import numpy as np
import os
import cv2
import threading
import Queue

def read_predicted_obj(fname):
	point_set = []
	with open(fname) as f:
		for line in f.readlines():
			line = line.strip('\n')
			c, x, y, z = line.split(' ')
			if c == 'v':
				x = np.float32(x)
				y = np.float32(y)
				z = np.float32(z)
				point_set.append([x, y, z])
			else:
				break
		f.close()
	assert (len(point_set) > 0)
	point_set = np.array(point_set).reshape(len(point_set), 3)
	return point_set

def read_ply(fname, with_normal=True):
	point_set = []
	with open(fname) as f:
		i = 0
		for line in f.readlines():
			i += 1
			if i > 12:
				line = line.strip('\n')
				x, y, z, nx, ny, nz = line.split(" ")
				x = np.float32(x)
				y = np.float32(y)
				z = np.float32(z)
				nx = np.float32(nx)
				ny = np.float32(ny)
				nz = np.float32(nz)
				point_set.append([x, y, z, nx, ny, nz])
		f.close()
	assert (len(point_set)>0)

	point_set = np.array(point_set).reshape(len(point_set), 6)
	if not with_normal:
		point_set = point_set[:, 0:3]

	return point_set

def read_mesh_feature_label(fname):
	pts_label = []
	with open(fname) as f:
		lines = f.readlines()
		for i in range(10, len(lines)):
			_, _, _, r, g, b = lines[i].strip('\n').split(' ')
			r = np.uint8(r)
			# g = np.uint8(g)
			b = np.uint8(b)
			if r == 255:
				pts_label.append(1)
			# elif b == 255:
			else:
				pts_label.append(0)
		f.close()
	assert (len(pts_label) == len(lines)-10)

	pts_label = np.array(pts_label, dtype=np.float32).reshape(len(pts_label), 1)
	return pts_label

def save_ply(points, fname, with_normal=True):
    points = np.array(points)
    nc = points.shape[-1]
    nv = points.shape[-2]

    mesh_writer = open(fname, 'wb')
    if with_normal is True or nc == 6:
        # begin ply header
        mesh_writer.write("ply" + '\n')
        mesh_writer.write("format ascii 1.0" + '\n')
        mesh_writer.write("element vertex " + str(nv) + '\n')
        mesh_writer.write("property float x" + '\n')
        mesh_writer.write("property float y" + '\n')
        mesh_writer.write("property float z" + '\n')
        mesh_writer.write("property float nx" + '\n')
        mesh_writer.write("property float ny" + '\n')
        mesh_writer.write("property float nz" + '\n')
        mesh_writer.write("element face 0" + '\n')
        mesh_writer.write("property list uchar int vertex_indices" + '\n')
        mesh_writer.write("end_header" + '\n')
        # end ply header

        for i in range(0, nv):
            mesh_writer.write(
                str(points[i][0]) + " " + str(points[i][1]) + " " + str(points[i][2]) + " " + str(points[i][3]) + " " +
                str(points[i][4]) + " " + str(points[i][5]) + '\n')
    else:
        # begin ply header
        mesh_writer.write("ply" + '\n')
        mesh_writer.write("format ascii 1.0" + '\n')
        mesh_writer.write("element vertex " + str(nv) + '\n')
        mesh_writer.write("property float x" + '\n')
        mesh_writer.write("property float y" + '\n')
        mesh_writer.write("property float z" + '\n')
        mesh_writer.write("element face 0" + '\n')
        mesh_writer.write("property list uchar int vertex_indices" + '\n')
        mesh_writer.write("end_header" + '\n')
        # end ply header

        for i in range(0, nv):
            mesh_writer.write(
                str(points[i][0]) + " " + str(points[i][1]) + " " + str(points[i][2]) + '\n')

    mesh_writer.close()

def jet_color(t):
    assert t >= 0
    if t >= 1:
        return np.array([255, 0, 0], dtype=np.uint8)

    l = 2. + np.sqrt(2.)
    t0 = 1./l
    t1 = (1. + np.sqrt(2.)) / l
    r = np.array([255, 0, 0], dtype=np.uint8)
    y = np.array([255, 255, 0], dtype=np.uint8)
    cy = np.array([0, 255, 255], dtype=np.uint8)
    b = np.array([0, 0, 255], dtype=np.uint8)

    rt = np.zeros(3, dtype=float)
    if t <= t0:
        s = 1-t/t0
        rt = s*b + (1-s)*cy
    elif t <= t1:
        s = 1-(t-t0)/(t1-t0)
        rt = s*cy + (1-s)*y
    else:
        s = 1-(t-t1)/(1-t1)
        rt = s*y + (1-s)*r

    return np.array(rt, dtype=np.uint8)


def save_label_mesh_with_color_ply(points, labels, faces, fname):
    nv = points.shape[-2]
    nf = faces.shape[-2]

    w_mes = open(fname, 'wb')

    w_mes.write("ply" + '\n')
    w_mes.write("format ascii 1.0" + '\n')
    w_mes.write("element vertex " + str(nv) + '\n')
    w_mes.write("property float x" + '\n')
    w_mes.write("property float y" + '\n')
    w_mes.write("property float z" + '\n')
    w_mes.write("property uchar red" + '\n')
    w_mes.write("property uchar green" + '\n')
    w_mes.write("property uchar blue" + '\n')
    w_mes.write("element face " + str(nf) + '\n')
    w_mes.write("property list uchar int vertex_indices" + '\n')
    w_mes.write("end_header" + '\n')

    for i in range(0, nv):
        color = jet_color(labels[i])
        w_mes.write(
            str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2]) + ' ' + str(color[0]) + ' ' + str(
                color[1]) + ' ' + str(color[2]) + '\n')
    for i in range(0, nf):
        w_mes.write('3 ' + str(faces[i][0]) + ' ' + str(faces[i][1]) + ' ' + str(faces[i][2]) + '\n')
    w_mes.close()

class DataFetcher(threading.Thread):
	def __init__(self, sket_root, pc_root, file_list):
		super(DataFetcher, self).__init__()
		self.sket_root = sket_root
		self.pc_root = pc_root
		self.stopped = False
		self.queue = Queue.Queue(64)

		self.sket_list = []
		with open(file_list, 'r') as f:
			while(True):
				line = f.readline().strip('\n')
				if not line:
					break
				self.sket_list.append(line)
		self.index = 0
		self.number = len(self.sket_list)

	def work(self, idx):
		sket_name = self.sket_list[idx]
		rand_val = np.random.randint(1, 11)
		img_front = cv2.resize(cv2.imread(os.path.join(self.sket_root, sket_name + '_0_sc.png')), (224, 224),
							   interpolation=cv2.INTER_CUBIC)
		if rand_val > 5:
			img_side = cv2.resize(cv2.imread(os.path.join(self.sket_root, sket_name + '_1_sc.png')), (224, 224),
								  interpolation=cv2.INTER_CUBIC)
		else:
			img_side = cv2.resize(cv2.imread(os.path.join(self.sket_root, sket_name + '_2_sc.png')), (224, 224),
								  interpolation=cv2.INTER_CUBIC)

		img_front = img_front.astype('float32') / 255.0
		img_side = img_side.astype('float32') / 255.0

		tmps = sket_name.split('_')
		mesh_name = tmps[0] + '_' + tmps[1] + '_' + tmps[2] + '.ply'
		pc = read_ply(os.path.join(self.pc_root, mesh_name))

		return img_front, img_side, pc, sket_name
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.sket_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()


if __name__ == '__main__':
	# file_list = sys.argv[1]
	data_root = '/media/administrator/Data/Sketch2Monster/'
	file_list = 'train_list_sel.txt'
	data = DataFetcher(os.path.join(data_root, 'sketch_sel'), os.path.join(data_root, 'pc_sel'),
					   os.path.join(data_root, 'mesh_sel_feat'), os.path.join(data_root, file_list))
	data.start()

	image, pc, feat_label, fname = data.fetch()
	print image.shape
	print(pc.shape)
	print feat_label.shape
	print fname
	print(np.array(np.where(feat_label==1)).shape)
	print(feat_label)
	data.stopped = True
