FROM maziyarnz/serverless:nightcore as nightcore
FROM maziyarnz/serverless:2.1.0 as pytorchDnn

WORKDIR /nightcore
COPY --from=nightcore /nightcore/launcher  /nightcore/launcher
COPY --from=nightcore /nightcore/worker/python  /nightcore/worker/python
COPY dnn_worker.py /nightcore/dnn_worker.py
ENV PYTHONPATH=/nightcore/worker/python

ENTRYPOINT [ "/nightcore/launcher", \
             "--fprocess=python3 /nightcore/dnn_worker.py", \
             "--fprocess_working_dir=/frontend", \
             "--fprocess_mode=go" ]
