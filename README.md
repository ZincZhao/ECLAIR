# install

```
pip install -e .
```

## how to use

### competence prediction

See tests/demo_multi.py

Input is a list of rchili json responses, see samples in the above demo.

Output is a list of competence levels corresponding to each resume, e.g., `['CRCI']`.

### acceptance prediction

See tests/demo_binary.py

Input is a list of rchili json responses with the applied job, see samples in the above demo.

Output is a list of decisions corresponding to each input, e.g., `['YES']`.
