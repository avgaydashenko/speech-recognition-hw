Speech recognition homework

Notes:
* Data for only single speaker is used (p239) from VCTK-Corpus as it has the most number of entries
* Label Error Rate should be considered as Character Error Rate since we don't have phomene marking in the dataset
* LER is calculated just the same as in the paper

Refs:
* paper: ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
* deepspeech: https://github.com/SeanNaren/deepspeech.pytorch
* tuto: http://sergeiturukin.com/2017/03/22/training-experiments.html
* ctc: https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding

Running:
`docker run -d -it -v `pwd`:/workspace/speech --name ctc-hw continuumio/miniconda3:latest`

`docker exec -it ctc-hw bash`

`cd speech/repos/warp-ctc/`

`mkdir build; cd build`

`apt update -y`

`apt install --no-install-recommends build-essential cmake make gcc`

`conda install pytorch-cpu torchvision -c pytorch`

`cmake ..`

`make`

`python setup.py install`

`cd ../build; cp libwarpctc.so /opt/conda/lib/`

`cd ../../; git clone https://github.com/SeanNaren/deepspeech.pytorch.git`

`exit`

`docker commit ctc-hw ctc-hw:latest`

`docker stop ctc-hw`

`docker run -d -it -p 8888:8888 -v `pwd`:/workspace/speech --name ctc-hw-new ctc-hw:latest`

`docker exec -it ctc-hw-new bash`

`conda install jupyter`

`jupyter notebook --ip=0.0.0.0 --no-browser --allow-root &`

`conda install pandas`

`conda install scipy`

`conda install scikit-learn`

`conda install matplotlib`

`conda install tqdm`

`pip install librosa`

`pip install python-levenshtein`

`docker commit ctc-hw-new ctc-hw-final:latest`

`docker run -d -it -p 8888:8888 -v `pwd`:/workspace/speech --shm-size 8G -m 3.5G --memory-swap -1 --cpus 4 --name ctc-hw-final ctc-hw-final:latest`

`docker exec -it ctc-hw-final bash`

`jupyter notebook --ip=0.0.0.0 --no-browser --allow-root &`
