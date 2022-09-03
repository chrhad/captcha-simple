#!/bin/bash
curl http://hr-testcases.s3.amazonaws.com/2587/assets/sampleCaptchas.zip --output sampleCaptchas.zip && \
    unzip sampleCaptchas.zip && rm -f sampleCaptchas.zip
