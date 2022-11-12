# himitsu
This is an official implementation of the paper "Addressing Segmentation Ambiguity in Neural Linguistic Steganography" (Accepted at AACL-IJCNLP2022).

## Setup
```
$ git clone https://github.com/jumon/himitsu.git
$ pip install .
```

## Example Usage
The following command encodes a secret message (a string of 0s and 1s) into cover text.
Given a `language` parameter (en, ja, ru, default: en), the corresponding GPT-2 model is used for encoding.
The parameter `prompt` is the context used to generate the cover text.
For other parameters, see the `encode.py` script.
```
$ python encode.py "010101011111010101101010" --prompt "Hi Bob." --language "en"

 It would appear there's nothing new.
```
To decode the secret message from the cover text, use the `decode.py` script.
Remember to add leading spaces to the cover text if there are any.
The optional parameters have the same meaning as in the `encode.py` script and the same parameter values need to be used.
```
$ python decode.py " It would appear there's nothing new." --prompt "Hi Bob." --language "en"

01010101111101010110101000
```
The decoded secret message should be the same as the original secret message except for a few trailing 0s.
