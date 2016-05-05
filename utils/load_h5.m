function d = load_h5(file_path)

file_path = [file_path '.hdf5'];

fid = H5F.open(file_path);

did = H5D.open(fid, 'dataset');
d.dataset = H5D.read(did);
d.dataset = d.dataset{1};
H5D.close(did);

did = H5D.open(fid, 'gt');
d.gt = H5D.read(did);
H5D.close(did);

did = H5D.open(fid, 'class');
d.class = H5D.read(did);
H5D.close(did);

did = H5D.open(fid, 'flip');
d.flip = H5D.read(did);
H5D.close(did);

did = H5D.open(fid, 'overlap');
d.overlap = H5D.read(did)';
H5D.close(did);

did = H5D.open(fid, 'boxes');
d.boxes = H5D.read(did)';
H5D.close(did);

did = H5D.open(fid, 'imagename');
d.imagename = H5D.read(did);
d.imagename = d.imagename{1};
H5D.close(did);

did = H5D.open(fid, 'feat');
d.feat = H5D.read(did)';
H5D.close(did);

H5F.close(fid);

% hdf5_path = file_path;
%	d.dataset = h5read(hdf5_path, '/dataset');
%	d.dataset = d.dataset{1};
%	d.gt = h5read(hdf5_path, '/gt');
%	d.class = h5read(hdf5_path, '/class');
%	d.flip = h5read(hdf5_path, '/flip');
%	d.overlap = h5read(hdf5_path, '/overlap')';
%	d.boxes = h5read(hdf5_path, '/boxes')';
%	d.imagename = h5read(hdf5_path, '/imagename');
%	d.imagename = d.imagename{1};
%	d.feat = h5read(hdf5_path, '/feat')';
