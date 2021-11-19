# install

```
pip install -e .
```

For GPU support, refer to [ELIT doc](https://github.com/emorynlp/elit/blob/main/docs/getting_started.md).

## how to use

### competence prediction

See tests/demo_multi.py

Input is a list of rchili json responses, see samples in the above demo.

Output is a list of competence levels corresponding to each resume, e.g., 

```
[
  {
    "best_prediction": [
      [
        "CRCI",
        0.629580557346344
      ]
    ],
    "predictions": {
      "NQ": 0.3691118359565735,
      "CRCI": 0.629580557346344,
      "CRCII": 0.00020453493925742805,
      "CRCIII": 0.00013402311014942825,
      "CRCIV": 0.0009691239101812243
    }
  }
]
```

### acceptance prediction

See tests/demo_binary.py

Input is a list of rchili json responses with the applied job, see samples in the above demo.

Output is a list of decisions corresponding to each input, e.g., 

```
[
  {
    "best_prediction": [
      [
        "YES",
        0.9916120767593384
      ]
    ],
    "predictions": {
      "NO": 0.008387885987758636,
      "YES": 0.9916120767593384
    }
  }
]
```

