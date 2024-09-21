import numpy as np

'''
the original is here https://www.pygame.org/wiki/OBJFileLoader
@2018-1-2 author chj
change for easy use
'''

class OBJ:
    def __init__(self, fdir, filename, swapyz=False):
        """ 
        加载并保存OBJ文件 
        """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        self.mtl=None
        material = None

        for line in open(fdir + filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue

            if values[0] == 'v':
                #v = map(float, values[1:4])
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                #v = map(float, values[1:4])
                v=[ float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                v = [float(x) for x in values[1:3]]
                self.texcoords.append(v)
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                #print(values[1])
                #self.mtl = MTL(fdir,values[1])
                self.mtl = [fdir,values[1]]
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

    def create_bbox(self):
        # self.vertices is not None
        ps=np.array(self.vertices)
        vmin=ps.min(axis=0)
        vmax=ps.max(axis=0)

        self.bbox_center=(vmax+vmin)/2
        self.bbox_half_r=np.max(vmax-vmin)/2



def find_vertices(faces, selected_vertices):
    """
    找到选定顶点周围的所有顶点
    faces:              所有面
    selected_vertices:  选定点
    """
    face_around = []                      # 所有包含了选中点的面的序号
    vertices_around = []                  # 选中点周围所有顶点
    # 记录包含选中点的面的序号
    for p in range(len(list(faces))):    
        for q in range(len( list(selected_vertices) )):
            if (selected_vertices[q]+1) in list(faces[p]):
                face_around.append(p)
                break    
    # 记录选中点+周围所有点，并去重
    for p in face_around:
        for q in range(3):
            vertices_around.append(faces[p, q])
    vertices_around = list(set(vertices_around))
    # 移除选中点
    for i in selected_vertices:
        vertices_around.remove(i+1)
    # 顶点索引恢复到从零开始
    for i in range(len(vertices_around)):
        vertices_around[i] -= 1
    
    return vertices_around



def Mov_corner(lmk_path_ori, lmk_path_fit, vertices):
    """
    返回包含移动信息的字典:
    Input:
        lmk_path_ori:   原图片的Landmark检测结果
        lmk_path_fit:   模型嵌入后的Landmark检测结果
        vertices:       模型所有顶点的数组
    Output:
        Mov:            嘴角点初始位移
        Corner_point:   嘴角点
        Lip_point:      嘴唇点
        Flag:           左或右
    """
    import numpy as np

    lmk_ori = np.load(lmk_path_ori)
    lmk_fit = np.load(lmk_path_fit)

    # 图片尺度landmark的误差
    mouth_ori_left = (lmk_ori[54] + lmk_ori[64]) / 2                               # 左边嘴角
    mouth_fit_left = (lmk_fit[54] + lmk_fit[64]) / 2
    lmk_loss_left = mouth_ori_left - mouth_fit_left
    lmk_loss_left[1] = -lmk_loss_left[1]                                           # 由于Y轴指向问题，y方向需翻转
    len_left = np.linalg.norm(lmk_loss_left)

    mouth_ori_right = (lmk_ori[48] + lmk_ori[60]) / 2                              # 右边嘴角
    mouth_fit_right = (lmk_fit[48] + lmk_fit[60]) / 2
    lmk_loss_right = mouth_ori_right - mouth_fit_right
    lmk_loss_right[1] = -lmk_loss_right[1]
    len_right = np.linalg.norm(lmk_loss_right)
    
    # 添加判断确定需要移动的顶点    (是否回拉如何确定？）
    if len_left >= len_right:
        lmk_loss = lmk_loss_left
        corner_point = 1730
        lip_point = 1826
        flag = 'left' 
    else:
        lmk_loss = lmk_loss_right
        corner_point = 2845
        lip_point = 2928
        flag = 'right'

    # 图片尺度鼻子长度
    nose_len_lmk = np.linalg.norm(lmk_fit[27]-lmk_fit[30])
    # 图片尺度鼻子宽度
    nose_width_lmk = np.linalg.norm(lmk_fit[31]-lmk_fit[35])
    # 图片尺度眼睛宽度
    eyes_left_lmk = np.linalg.norm(lmk_fit[36]-lmk_fit[39])
    eyes_right_lmk = np.linalg.norm(lmk_fit[42]-lmk_fit[45])
    eyes_width_lmk = (eyes_left_lmk + eyes_right_lmk) / 2


    # obj尺度鼻子长度
    nose_up_obj, nose_down_obj = vertices[3516][:2], vertices[3564][:2]        # 只取XY坐标
    nose_len_obj = np.linalg.norm(nose_up_obj - nose_down_obj)
    # obj尺度鼻子宽度
    nose_left_obj, nose_right_obj = vertices[1613][:2], vertices[477][:2]
    nose_width_obj = np.linalg.norm(nose_left_obj - nose_right_obj)
    # obj尺度眼睛宽度
    eyes_left_left_obj, eyes_left_right_obj = vertices[2437][:2], vertices[3619][:2]
    eyes_right_left_obj, eyes_right_right_obj = vertices[3827][:2], vertices[1146][:2]
    eyes_left_obj = np.linalg.norm(eyes_left_left_obj- eyes_left_right_obj)
    eyes_right_obj = np.linalg.norm(eyes_right_left_obj - eyes_right_right_obj)
    eyes_width_obj = (eyes_left_obj + eyes_right_obj) / 2


    # 计算嘴角的初始Mov
    scale = ((nose_len_obj / nose_len_lmk) + (nose_width_obj / nose_width_lmk) + (eyes_width_obj / eyes_width_lmk)) / 3
    Mov = scale * lmk_loss
    Mov = Mov * 2                                               # 增大（系数怎么设定呢？）

    X_Mov = Mov[0] ; Y_Mov = Mov[1]
    Z_Mov = - (abs(X_Mov) + abs(Y_Mov)) / 2                     # 拟定的Z方向位移（怎么定呢？）
    Mov = np.array((X_Mov, Y_Mov, Z_Mov))                       # 初始位移向量

    # 字典记录移动信息
    Mov_info ={
        'Mov': Mov,
        'Corner_point':corner_point,
        'Lip_point': lip_point,
        'Flag':flag,
    }

    return Mov_info



def damping_mov(N, Mov, round, vertices, position):
    """
    衰减位移算法            (后续优化函数结构)
    N:          迭代次数
    Mov:        初始位移
    round:      包含所有分层点的列表
    vertices:   文件所有顶点
    position:   'corner' or 'lip'
    """
    import math

    Mov_Ori = Mov / N      # 初始点步长

    for p in range(N):
        for q in range(len(round)):
            cur = round[q]

            if position == 'corner':
                Mov_Step_X = Mov_Ori[0] * math.cos(q * math.pi / (2 * len(round))) * (len(round) - q) / len(round)          # 余弦形式 * 线性衰减因子
                Mov_Step_Y = Mov_Ori[1] * math.cos(q * math.pi / (2 * len(round))) * (len(round) - q) / len(round)
                Mov_Step_Z = Mov_Ori[2] * math.cos(q * math.pi / (len(round) - 1)) * (len(round) - q) / len(round)
            
            elif position == 'lip':
                Mov_Step_X = Mov_Ori[0] * math.cos(q * math.pi / (2 * len(round))) * (len(round) - q) / len(round)
                Mov_Step_Y = Mov_Ori[1] * math.cos(q * math.pi / (2 * len(round))) * (len(round) - q) / len(round)
                Mov_Step_Z = Mov_Ori[2] * (len(round) -  q) / len(round)          

            Mov_Step = np.array([Mov_Step_X, Mov_Step_Y, Mov_Step_Z])
        
            for i in range(len(cur)):
                vertices[cur[i]] += Mov_Step


def save_obj(file_path, vertices, faces):
    """
    保存obj
    文件
    file_path:  保存文件的路径
    vertices:   矫正后的顶点数组
    faces:      矫正后的面数组
    """
    vertices_list = []
    faces_list =[]

    for p in range(len(list(vertices))):
        cur = list(vertices[p])
        cur.insert(0, 'v')
        vertices_list.append(cur)

    for p in range(len(list(faces))):
        cur = list(faces[p])
        cur.insert(0, 'f')
        faces_list.append(cur)

    with open(file_path, 'w') as f:
        for line in vertices_list:
            f.write(str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + '\n')
        for line in faces_list:
            f.write(str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + '\n')
    
    print('The file has been saved to: ', file_path)
    
