VOCdevkit = './data/VOCdevkit2007/';
tmp = pwd;
cd(VOCdevkit);
addpath([cd '/VOCcode']);
VOCinit;
cd(tmp);