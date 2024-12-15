# dml-tracer
A tool that traces data flows in Distributed Machine Learning. The tracer collects data flow information such as size and type of the flow, source and destination, and flow dependencies.

# Example: GPT2

## Install required dependencies
```
pip install -r requirements.txt
```

## Usage
```
cd dml_tracer
make
<should generated dml_tracer.so file>
```
```
cd <root of this repo>
LD_PRELOAD=dml_tracer/dml_tracer.so python GPT2Dist.py
```
