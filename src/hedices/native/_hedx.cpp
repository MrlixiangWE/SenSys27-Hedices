// Native execution channel of the Branch (Alg. 2, process-pool branch of @NoRely).
//
// One Channel couples the Branch with one pre-started executor process through a POSIX
// shared-memory segment: the request and the reply are copied into the segment and a futex on
// the segment's state word hands it over, so a call costs two memcpy and two wake-ups instead
// of the pipes, feeder threads and pickled envelopes of a multiprocessing pool.  Every wait
// releases the GIL, so the Branch's RPC threads keep running while a stage executes elsewhere.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <linux/futex.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

enum : uint32_t { IDLE = 0, REQ = 1, REP = 2, CLOSED = 3 };

struct Header {
    std::atomic<uint32_t> state;
    uint32_t pad;
    uint64_t len;
    uint64_t cap;
};

long futex(std::atomic<uint32_t> *addr, int op, uint32_t val, const timespec *ts) {
    return syscall(SYS_futex, reinterpret_cast<uint32_t *>(addr), op, val, ts, nullptr, 0);
}

void wake(std::atomic<uint32_t> *addr) { futex(addr, FUTEX_WAKE, INT_MAX, nullptr); }

// Wait until *addr == want.  A short spin covers hand-overs that complete within microseconds;
// then futex sleeps.  Returns false on timeout (timeout_ms < 0: wait forever) or CLOSED.
bool await_state(std::atomic<uint32_t> *addr, uint32_t want, long timeout_ms) {
    for (int i = 0; i < 2000; ++i) {
        uint32_t s = addr->load(std::memory_order_acquire);
        if (s == want) return true;
        if (s == CLOSED) return false;
    }
    timespec slice{0, 50 * 1000 * 1000};
    long waited = 0;
    for (;;) {
        uint32_t s = addr->load(std::memory_order_acquire);
        if (s == want) return true;
        if (s == CLOSED) return false;
        if (timeout_ms >= 0 && waited >= timeout_ms) return false;
        futex(addr, FUTEX_WAIT, s, &slice);
        waited += 50;
    }
}

struct Channel {
    PyObject_HEAD
    Header *hdr;
    char *data;
    size_t map_len;
    char name[128];
    int owner;
};

void channel_dealloc(Channel *self) {
    if (self->hdr) munmap(self->hdr, self->map_len);
    if (self->owner) shm_unlink(self->name);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

int channel_init(Channel *self, PyObject *args, PyObject *kw) {
    static const char *kws[] = {"name", "capacity", "create", nullptr};
    const char *name;
    unsigned long long cap = 64ull << 20;
    int create = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "s|Kp", const_cast<char **>(kws), &name, &cap, &create))
        return -1;
    if (strlen(name) >= sizeof(self->name)) {
        PyErr_SetString(PyExc_ValueError, "channel name too long");
        return -1;
    }
    strcpy(self->name, name);
    int fd = shm_open(name, create ? (O_CREAT | O_EXCL | O_RDWR) : O_RDWR, 0600);
    if (fd < 0) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, name);
        return -1;
    }
    size_t len = sizeof(Header) + cap;
    if (create && ftruncate(fd, static_cast<off_t>(len)) != 0) {
        close(fd);
        shm_unlink(name);
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }
    if (!create) {
        struct stat st;
        fstat(fd, &st);
        len = static_cast<size_t>(st.st_size);
    }
    void *p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        if (create) shm_unlink(name);
        PyErr_SetFromErrno(PyExc_OSError);
        return -1;
    }
    self->hdr = static_cast<Header *>(p);
    self->data = static_cast<char *>(p) + sizeof(Header);
    self->map_len = len;
    self->owner = create;
    if (create) {
        self->hdr->cap = cap;
        self->hdr->len = 0;
        self->hdr->state.store(IDLE, std::memory_order_release);
    }
    return 0;
}

bool put(Channel *self, Py_buffer *buf) {
    if (static_cast<uint64_t>(buf->len) > self->hdr->cap) {
        PyErr_Format(PyExc_OverflowError, "message of %zd B exceeds the channel's %llu B",
                     buf->len, static_cast<unsigned long long>(self->hdr->cap));
        return false;
    }
    memcpy(self->data, buf->buf, static_cast<size_t>(buf->len));
    self->hdr->len = static_cast<uint64_t>(buf->len);
    return true;
}

// Branch side: send a request, wait for the reply.  pid: the executor, checked while waiting.
PyObject *channel_call(Channel *self, PyObject *args) {
    Py_buffer buf;
    int pid = 0;
    if (!PyArg_ParseTuple(args, "y*|i", &buf, &pid)) return nullptr;
    if (self->hdr->state.load(std::memory_order_acquire) != IDLE) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_RuntimeError, "channel busy");
        return nullptr;
    }
    bool ok = put(self, &buf);
    PyBuffer_Release(&buf);
    if (!ok) return nullptr;
    bool done = false, dead = false;
    Py_BEGIN_ALLOW_THREADS
    self->hdr->state.store(REQ, std::memory_order_release);
    wake(&self->hdr->state);
    for (;;) {
        if (await_state(&self->hdr->state, REP, 500)) { done = true; break; }
        if (self->hdr->state.load(std::memory_order_acquire) == CLOSED) break;
        if (pid > 0 && kill(pid, 0) != 0 && errno == ESRCH) { dead = true; break; }
    }
    Py_END_ALLOW_THREADS
    if (!done) {
        PyErr_SetString(PyExc_ChildProcessError, dead ? "executor died" : "channel closed");
        return nullptr;
    }
    PyObject *out = PyBytes_FromStringAndSize(self->data, static_cast<Py_ssize_t>(self->hdr->len));
    self->hdr->state.store(IDLE, std::memory_order_release);
    return out;
}

// Executor side: block (without the GIL) for the next request; None once the channel closes.
PyObject *channel_recv(Channel *self, PyObject *) {
    bool ok;
    Py_BEGIN_ALLOW_THREADS
    ok = await_state(&self->hdr->state, REQ, -1);
    Py_END_ALLOW_THREADS
    if (!ok) Py_RETURN_NONE;
    return PyBytes_FromStringAndSize(self->data, static_cast<Py_ssize_t>(self->hdr->len));
}

PyObject *channel_reply(Channel *self, PyObject *args) {
    Py_buffer buf;
    if (!PyArg_ParseTuple(args, "y*", &buf)) return nullptr;
    bool ok = put(self, &buf);
    PyBuffer_Release(&buf);
    if (!ok) return nullptr;
    self->hdr->state.store(REP, std::memory_order_release);
    wake(&self->hdr->state);
    Py_RETURN_NONE;
}

PyObject *channel_close(Channel *self, PyObject *) {
    self->hdr->state.store(CLOSED, std::memory_order_release);
    wake(&self->hdr->state);
    Py_RETURN_NONE;
}

PyMethodDef channel_methods[] = {
    {"call", reinterpret_cast<PyCFunction>(channel_call), METH_VARARGS, "call(request[, pid]) -> reply"},
    {"recv", reinterpret_cast<PyCFunction>(channel_recv), METH_NOARGS, "recv() -> request | None"},
    {"reply", reinterpret_cast<PyCFunction>(channel_reply), METH_VARARGS, "reply(data)"},
    {"close", reinterpret_cast<PyCFunction>(channel_close), METH_NOARGS, "close the channel"},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject ChannelType = {PyVarObject_HEAD_INIT(nullptr, 0)};

// The executor dies with its Branch (a killed worker must not leave executors behind).
PyObject *die_with_parent(PyObject *, PyObject *) {
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    if (getppid() == 1) _exit(1);
    Py_RETURN_NONE;
}

PyMethodDef module_methods[] = {
    {"die_with_parent", die_with_parent, METH_NOARGS, "SIGKILL this process when its parent exits"},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "_hedx", "Hedices native execution channel", -1,
                      module_methods};

}  // namespace

PyMODINIT_FUNC PyInit__hedx(void) {
    ChannelType.tp_name = "hedices.native._hedx.Channel";
    ChannelType.tp_basicsize = sizeof(Channel);
    ChannelType.tp_flags = Py_TPFLAGS_DEFAULT;
    ChannelType.tp_new = PyType_GenericNew;
    ChannelType.tp_init = reinterpret_cast<initproc>(channel_init);
    ChannelType.tp_dealloc = reinterpret_cast<destructor>(channel_dealloc);
    ChannelType.tp_methods = channel_methods;
    if (PyType_Ready(&ChannelType) < 0) return nullptr;
    PyObject *m = PyModule_Create(&module);
    if (!m) return nullptr;
    Py_INCREF(&ChannelType);
    PyModule_AddObject(m, "Channel", reinterpret_cast<PyObject *>(&ChannelType));
    return m;
}
