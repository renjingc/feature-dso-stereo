cd build
make -j8
cd ..

./build/bin/dso_dataset \
files=/media/ren/99146341-07be-4601-9682-0539688db03f/我的数据集/ourdata/data16/img \
calib=/media/ren/99146341-07be-4601-9682-0539688db03f/我的数据集/ourdata/data16/calibLeft.txt \
calibRight=/media/ren/99146341-07be-4601-9682-0539688db03f/我的数据集/ourdata/data16/calibRight.txt \
vocab=/home/ren/work/dso_semantic/vocab/ORBvoc.bin \
gtPath= \
preset=3 \
mode=1