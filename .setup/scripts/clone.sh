#!/bin/bash

git clone git@github.com:ijapesigan/fitAutoReg.git
rm -rf "$PWD.git"
mv fitAutoReg/.git "$PWD"
rm -rf fitAutoReg
