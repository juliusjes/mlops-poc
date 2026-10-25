import sys

_TRACE = {}
_PATCHED = set()


def enable_tracing(obj, targets): #TODO: does not work multiple of the same drift detector
    """
    Enable variable tracing for Evidently Drift methods
    """
    cls = obj.__class__

    _TRACE[obj.get_fingerprint()] = {
        "targets": targets,
        "locals": {}
    }

    if cls in _PATCHED:
        return obj

    old_call = cls.__call__
    target_code = old_call.__code__

    def traced_call(self, *args, **kwargs):

        # Fast path
        if self.get_fingerprint() not in _TRACE:
            return old_call(self, *args, **kwargs)

        captured = {}
        target_frame = None

        def tracer(frame, event, arg):
            nonlocal target_frame
            
            if (
                event == "call"
                and frame.f_code is target_code
            ):
                target_frame = frame

            elif (
                event == "return"
                and frame is target_frame
            ):
                for t in _TRACE[self.get_fingerprint()]["targets"]:
                    captured[t] = frame.f_locals.get(t)

            return tracer

        old = sys.gettrace()
        sys.settrace(tracer)

        try:
            return old_call(self, *args, **kwargs)
        finally:
            sys.settrace(old)
            _TRACE[self.get_fingerprint()]["locals"] = captured

    cls.__call__ = traced_call
    _PATCHED.add(cls)

    return obj


if __name__=="__main__":
    import inspect
    from inference_server.detectors.conformal import Conformal
    from inference_server.detectors.evd import MMDDriftMethod
    import random
    import string
    import pandas as pd

    data = {}

    for i in range(128):
        col = "".join(
            random.choices(
                string.ascii_letters + "åäö",
                k=8
            )
        )
        data[col] = list(range(i - 100, i))

    data = pd.DataFrame(data)


    detector = enable_tracing(Conformal(), targets=["thold"])

    detector(data, data)

    print(_TRACE[id(detector)]["locals"])

    # benchmark
    import time
    mmd_trace = enable_tracing(MMDDriftMethod(bootstrap=True), targets=["mmd2u"])
    conf = MMDDriftMethod(bootstrap=True)

    times_mmd = []
    time_conf = []

    for j in range(10):
        t0 = time.time()
        mmd_trace(data, data)
        times_mmd.append(time.time() - t0)



        t0 = time.time()
        conf(data, data)
        time_conf.append(time.time() - t0)

    print(f"mmd: {sum(times_mmd) / 10}")
    print(f"conf: {sum(time_conf) / 10}")