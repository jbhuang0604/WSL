function [] = hdf5_dir_to_mat_dir(hdf5_dir_path, mat_dir_path, quiet, skip_done)
	assert(logical(exist(hdf5_dir_path, 'dir')));
	if ~exist('quiet', 'var')
		quiet = false;
	end
	if ~exist('skip_done', 'var')
		skip_done = true;
	end
	if ~exist(mat_dir_path, 'dir')
		mkdir(mat_dir_path);
	end
	files = dir(sprintf('%s/*.hdf5', hdf5_dir_path));
	parfor i = 1:length(files)
		[~, name, ~] = fileparts(files(i).name);
		hdf5_path = sprintf('%s/%s.hdf5', hdf5_dir_path, name);
		mat_path = sprintf('%s/%s.mat', mat_dir_path, name);
		if exist(mat_path, 'file') && skip_done
			continue;
		end
		hdf5_to_mat(hdf5_path, mat_path);
		if ~quiet
			fprintf('(%d/%d) Converted %s to %s\n', i, length(files), hdf5_path, mat_path);
		end
	end
end

function [] = hdf5_to_mat(hdf5_path, mat_path)
	x = hdf5_to_struct(hdf5_path);
	save(mat_path, '-struct', 'x');
end

function x = hdf5_to_struct(hdf5_path)
	x.dataset = h5read(hdf5_path, '/dataset');
	x.dataset = x.dataset{1};
	x.gt = h5read(hdf5_path, '/gt');
	x.class = h5read(hdf5_path, '/class');
	x.flip = h5read(hdf5_path, '/flip');
	x.overlap = h5read(hdf5_path, '/overlap')';
	x.boxes = h5read(hdf5_path, '/boxes')';
	x.imagename = h5read(hdf5_path, '/imagename');
	x.imagename = x.imagename{1};
	x.feat = h5read(hdf5_path, '/feat')';
end
