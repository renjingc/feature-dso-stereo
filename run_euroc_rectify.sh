#!/usr/bin/env bash

cd build
make -j8
cd ..

./build/bin/euroc_rectify /media/ren/99146341-07be-4601-9682-0539688db03f/我的数据集/ourdata/data16
