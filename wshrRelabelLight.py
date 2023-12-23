from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.io import loadmat
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# give the preset classification of variables
group_names_list = ["mechanism", "power", "control", "external", "recorder", "unclassified"]
group_lens_dict = {}

var_group_mechanism = ["AIL_1", "AIL_2", "FLAP", "ELEV_1", "ELEV_2", "RUDD", "SPL_1", "SPL_2", "SPLG", "SPLY", "ABRK", "BPGR_1", "BPGR_2", "BPYR_1", "BPYR_2", "MSQT_1", "MSQT_2", "NSQT", "BLV", "CALT", "PACK", "WOW", 
                       "AOA1", "AOA2", "GLS", "PTCH", "ROLL", 
                       "TH", "MH", "TAS", "CASM", "GS", "IVV",
                       "VRTG", "LATG", "LONG", "FPAC", "CTAC"]
var_group_power = ["N2_1", "N2_2", "N2_3", "N2_4",
                   "ECYC_1", "ECYC_2", "ECYC_3", "ECYC_4", "EHRS_1", "EHRS_2", "EHRS_3", "EHRS_4", "VIB_1", "VIB_2", "VIB_3", "VIB_4", "FADS", "HYDG", "HYDY",
                   "N1_1", "N1_2", "N1_3", "N1_4", "N1T", "FF_1", "FF_2", "FF_3", "FF_4", "FQTY_1", "FQTY_2", "FQTY_3", "FQTY_4", "OIP_1", "OIP_2", "OIP_3", "OIP_4", "OIT_1", "OIT_2", "OIT_3", "OIT_4", "OIPL", "EGT_1", "EGT_2", "EGT_3", "EGT_4",
                   "LGDN", "LGUP"]
var_group_control = ["CRSS", "HDGS", "A_T", "APFD", "DFGS", "FGC3", "PUSH", "PTRM", "TCAS",
                     "ILSF", "RUDP", "CCPC", "CCPF", "CWPC", "CWPF", "PLA_1", "PLA_2", "PLA_3", "PLA_4",
                     "SNAP", "TMODE", "EAI", "TAI", "WAI_1", "WAI_2", 
                     "APUF", "FADF", "FIRE_1", "FIRE_2", "FIRE_3", "FIRE_4", "GPWS", "MW", "POVT", "SHKR", "SMOK", "TOCW"]
var_group_external = ["ALT", "ALTR", "WS", "WD", "PI", "PS", "PT", "SAT", "TAT",
                      "DA", "TRK", "TRKM", "LOC", "LATP", "LONP"]
var_group_recorder = ["DWPT", "PH", 
                     "ACMT", "FRMC", "GMT_HOUR", "GMT_MINUTE", "GMT_SEC"]
var_group_unclassified = ["ATEN", "EVNT", "HF1", "HF2", "VHF1", "VHF2", "VHF3", "LMOD", "VMODE", "MACH", "MNS", "MRK", "N1C", "N1CO", "SMKB", "VAR_1107", "VAR_2670", "VAR_5107", "VAR_6670"]

var_groups_dict = {"mechanism": var_group_mechanism, "power": var_group_power, "control": var_group_control, "external": var_group_external, "recorder": var_group_recorder, "unclassified": var_group_unclassified}
# for group_name, var_group in var_groups_dict.items():
#     group_lens_dict[group_name] = len(var_group)
#     print(f"{group_name}: {len(var_group)}")
# print(f"\n{sum(group_lens_dict.values())} variables in total")

# 查找给定总序数对应的变量名称
def find_var_name(idx, var_dict):
    count = 0
    group_lens_dict = {}
    for group_name, var_group in var_dict.items():
        group_lens_dict[group_name] = len(var_group)
    for group_name, var_group in var_dict.items():
        if count + group_lens_dict[group_name] > idx:
            return group_name, var_group[idx - count]
        else:
            count += group_lens_dict[group_name]

# 查找给定变量名称对应的总序数
def find_var_idx(var_name, var_dict):
    count = 0
    for var_list in var_dict.values():
        if var_name in var_list:
            count += var_list.index(var_name)
            return(count)
        else:
            count += len(var_list)

def dataSampling(var_data, var_rate, wshr_data):
    # 对每个变量按照rate进行下采样或过采样，对长为n+1的数据，抓取前n个全变量为输入，后n个有缺变量为输出
    if var_rate == 1:
        sampling_data = var_data
    else: # 进行重采样
        sampling_data = [var_data[i] for i in np.linspace(0, len(var_data)- 1, len(wshr_data), dtype=int)]
    # # 将采样数据进行min_max归一化
    # if (np.max(sampling_data) - np.min(sampling_data)) > 1e-5:
    #     sampling_data = (sampling_data - np.min(sampling_data)) / (np.max(sampling_data) - np.min(sampling_data))
    # else:
    #     sampling_data = sampling_data
    return sampling_data

def dataConstruct(work_folder_path, work_mat_name, variable_list, normalized = False, is_all_variable = False):
    # 读取工作mat
    work_mat = loadmat(os.path.join(work_folder_path, work_mat_name))

    # 初始化采样数据list
    sampling_data_list = []
    wshr_data = work_mat["WSHR"][0][0][0]

    if not is_all_variable:
        # 提取variable_list变量并进行重采样
        for var_name in variable_list:
            var_data, var_rate = work_mat[var_name][0][0][0], work_mat[var_name][0][0][1][0][0]
            sampling_data = dataSampling(var_data, var_rate, wshr_data)
            # print(np.max(sampling_data), np.min(sampling_data))
            sampling_data_list.append(sampling_data)
    else:
        for var_list in var_groups_dict.values():
            for var_name in var_list:
                var_data, var_rate = work_mat[var_name][0][0][0], work_mat[var_name][0][0][1][0][0]
                sampling_data = dataSampling(var_data, var_rate, wshr_data)
                sampling_data_list.append(sampling_data)
        sampling_data_list = np.squeeze(np.array(sampling_data_list))

    # 生成解释变量X和分类变量Y
    X = np.squeeze(np.array(sampling_data_list)).T
    Y = np.array(wshr_data).reshape(-1,1)
    if normalized:
        # 数据标准化
        s_scaler = StandardScaler()
        X = s_scaler.fit_transform(X)

    wshr_class_idx = [np.where(Y == 0)[0], np.where(Y == 1)[0]]
    # print(X.shape, Y.shape)
    # print(f"Wind Shear Warns at time {wshr_class_idx[0]}")
    return X, Y

def ALTRmetrics(folder_name, mat_name, normalized=False):
    # 构建数据集
    variable_list = ['ALT', 'ALTR', 'PT']
    X, Y = dataConstruct(folder_name, mat_name, variable_list)

    # 以60s为滑动窗口，计算海拔的变化值
    ALT_diff = X[60:, 0] - X[:-60, 0]

    # 比较ALTR和分钟滑动的ALT
    '''
    ALTR相当于在这一秒预测接下来一分钟的升降率（基于下一秒的步差），ALT分钟滑动相当于计算接下来一分钟实际发生的升降率
    前者反映瞬间的速度方向，后者反映一分钟内的平均位移水平（是计划上升还是巡航还是计划下降）
    二者的差分反映这一秒的升降率是否在计划内
    滑动的窗口大小实际上可以调节，但是考虑到单位是英尺/分钟，选取了1分钟做窗口
    '''
    ALT_diff_extend = [0 if i-60 < 0 else ALT_diff[i-60] for i in range(len(X[:, 1]))]
    # ALT_diff_extend = [0 if i-1 < 0 else ALT_diff[i-1]*60 for i in range(len(X[:, 1]))]
    ALTR_residual = np.array(ALT_diff_extend) - X[:, 1]

    # 计算ALTR相较于分钟滑动ALT的残差
    # ALT_stable_time = np.where(np.abs(ALT_diff) <= 30)[0]
    ALT_stable_time = []
    wshr_class_idx = [np.where(Y == 0)[0], np.where(Y == 1)[0]]
    # print(np.min(ALT_stable_time), np.max(ALT_stable_time))
    ALTR_residual = np.array(ALT_diff_extend) - X[:, 1]
    ALTR_overlimit = [ALTR_residual[i] if abs(ALTR_residual[i])>500 and i not in ALT_stable_time else None for i in range(len(ALTR_residual))]
    ALTR_IC_idx_list = np.where(np.array(ALTR_overlimit) == None)[0]
    ALTR_OOC_idx_list = np.where(np.array(ALTR_overlimit) != None)[0]
    # print(f"Altitute varying rate out of control at time: {ALTR_OOC_idx_list}")

    return ALTR_OOC_idx_list

def PTCHmetrics(folder_name, mat_name, normalized=False):
    # 构建数据集
    variable_list = ['PTCH', 'AOA1', 'AOA2', 'TH'] 
    X, Y = dataConstruct(folder_name, mat_name, variable_list, normalized)

    # 打印俯仰姿态绝对值超过5度的时刻
    PTCH_IC_idx_list = np.where(np.abs(X[:, 0]) <= 5)[0]
    PTCH_OOC_idx_list = np.where(np.abs(X[:, 0]) > 5)[0]
    # print(f"Pitch angle out of control at time {PTCH_OOC_idx_list}")

    return PTCH_OOC_idx_list

def crossValidate(folder_name, mat_name, variable_list, normalized=False):
    # 构建数据集
    ALTR_OOC_idx_list = ALTRmetrics(folder_name, mat_name)
    PTCH_OOC_idx_list = PTCHmetrics(folder_name, mat_name)
    X, Y = dataConstruct(folder_name, mat_name, variable_list, normalized)
    wshr_class_idx = [np.where(Y == 0)[0], np.where(Y == 1)[0]]

    # 获取两判据OOC的交集
    cross_OOC_idx_list = np.sort(list(set(ALTR_OOC_idx_list) & set(PTCH_OOC_idx_list)))
    cross_IC_idx_list = [time for time in range(len(Y)) if time not in cross_OOC_idx_list]
    # print(f'Possible WSHR time: {cross_OOC_idx_list}')

    # 裁取交叉判定可能存在异常的(ALTR, PTCH)的数据集
    if len(cross_OOC_idx_list) > 0:
        X_cross = X[cross_OOC_idx_list, :2]
        X_cross_scaled = StandardScaler().fit_transform(X_cross)

        # # 对于crossing的(ALTR, PTCH)进行K-means++聚类
        # kmeans =  KMeans(n_clusters=6, random_state=42)
        # kmeans.fit(X_cross)
        # kmeans_centers = kmeans.cluster_centers_
        # kmeans_labels = kmeans.labels_

        # # 对于crossing的(ALTR, PTCH)进行DBSCAN聚类
        # db = DBSCAN(eps=0.3, min_samples=2).fit(X_cross_scaled)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # db_labels = db.labels_
        # # print(db_labels)
        # # for i in np.unique(db_labels):
        # #     print(f'Cluster {i}: {np.where(X[:, :2] == X_cross[db_labels == i])}')

        # # 绘制甘特图
        # fig, ax = plt.subplots(figsize=(20, 5))
        # ax.broken_barh([(time, 1) for time in ALTR_OOC_idx_list], (0-0.4, 0.8), label='ALTR out of control')
        # ax.broken_barh([(time, 1) for time in PTCH_OOC_idx_list], (1-0.4, 0.8), label='PTCH out of control')
        # ax.broken_barh([(time, 1) for time in cross_OOC_idx_list], (2-0.4, 0.8), facecolors = 'orange', label='Crossing OOC time')
        # ax.broken_barh([(time, 1) for time in wshr_class_idx[0]], (3-0.4, 0.8), facecolors = 'red', label='Given WSHR time')
        # ax.set_yticks(range(4))
        # ax.set_yticklabels(['ALTR OOC', "PTCH OOC", 'Crossing OOC', 'Given WSHR'])
        # plt.xlabel('time (s)')
        # plt.show()

        # # # 绘制ALTR和PTCH的散点图及kmneas聚类椭圆形式图
        # # for i in range(2):
        # #     plt.figure()
        # #     if i == 0:
        # #         plt.scatter(X[cross_IC_idx_list, 0], X[cross_IC_idx_list, 1], c='green', s = 10, label='Crossing IC')
        # #     plt.scatter(X[cross_OOC_idx_list, 0], X[cross_OOC_idx_list, 1], c='orange', s = 10, label='Crossing OOC')
        # #     plt.scatter(X[wshr_class_idx[0], 0], X[wshr_class_idx[0], 1], c='red', s = 10, label='WSHR = 0')
        # #     for i in range(len(kmeans_centers)):
        # #         covariances = np.cov(X_cross[kmeans_labels == i].T)
        # #         eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        # #         angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        # #         width, height = 3 * np.sqrt(eigenvalues)
        # #         ell = Ellipse(xy=kmeans_centers[i], width=width, height=height, angle=angle, edgecolor='blue', lw=2, facecolor='none')
        # #         plt.gca().add_patch(ell)
        # #     plt.xlabel(variable_list[0])
        # #     plt.ylabel(variable_list[1])
        # #     if i == 0:
        # #         plt.title('Crossing in-control & out-of-control points w.r.t (ALTR, PTCH)')
        # #     else:
        # #         plt.title('Crossing out-of-control points w.r.t (ALTR, PTCH)')
        # #     plt.legend()
        # #     plt.show()

        # # 绘制ALTR和PTCH的散点图及DBSCAN聚类椭圆形式图
        # for i in range(2):
        #     plt.figure()
        #     if i == 0:
        #         plt.scatter(X[cross_IC_idx_list, 0], X[cross_IC_idx_list, 1], c='green', s = 10, label='Crossing IC')
        #     plt.scatter(X[cross_OOC_idx_list, 0], X[cross_OOC_idx_list, 1], c='orange', s = 10, label='Crossing OOC')
        #     plt.scatter(X[wshr_class_idx[0], 0], X[wshr_class_idx[0], 1], c='red', s = 10, label='WSHR = 0')
        #     for i in range(len(np.unique(db_labels))):
        #         covariances = np.cov(X_cross[db_labels == i].T)
        #         eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        #         angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        #         width, height = 3 * np.sqrt(eigenvalues)
        #         ell = Ellipse(xy=np.mean(X_cross[db_labels == i], axis=0), width=width, height=height, angle=angle, edgecolor='blue', lw=2, facecolor='none')
        #         plt.gca().add_patch(ell)
        #     plt.xlabel(variable_list[0])
        #     plt.ylabel(variable_list[1])
        #     if i == 0:
        #         plt.title('Crossing in-control & out-of-control points w.r.t (ALTR, PTCH)')
        #     else:
        #         plt.title('Crossing out-of-control points w.r.t (ALTR, PTCH)')
        #     plt.legend()
        #     plt.show()

        # # 绘制三维散点图
        # fig = plt.figure(figsize=(5, 5))
        # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(X[wshr_class_idx[1], 0], X[wshr_class_idx[1], 1], X[wshr_class_idx[1], 2], c='green', s = 10, label='WSHR = 1')
        # ax.scatter(X[cross_IC_idx_list, 0], X[cross_IC_idx_list, 1], X[cross_IC_idx_list, 2], c='green', s = 10, label='Crossing IC')
        # ax.scatter(X[cross_OOC_idx_list, 0], X[cross_OOC_idx_list, 1], X[cross_OOC_idx_list, 2], c='orange', s = 10, label = 'Crossing OOC')
        # # ax.scatter(X[ALTR_OOC_idx_list, 0], X[ALTR_OOC_idx_list, 1], X[ALTR_OOC_idx_list, 2], c='pink', s = 10, label='ALTR out of control')
        # # ax.scatter(X[PTCH_OOC_idx_list, 0], X[PTCH_OOC_idx_list, 1], X[PTCH_OOC_idx_list, 2], c='blue', s = 10, label='PTCH out of control')
        # ax.scatter(X[wshr_class_idx[0], 0], X[wshr_class_idx[0], 1], X[wshr_class_idx[0], 2], c='red', s = 10, label='WSHR = 0')
        # ax.set_xlabel(variable_list[0])
        # ax.set_ylabel(variable_list[1])
        # ax.set_zlabel(variable_list[2])
        # ax.legend()
        # plt.tight_layout()
        # plt.show()
    else:
        cross_OOC_idx_list = []

    return cross_OOC_idx_list

# crossValidate(download_folder_paths[0], os.listdir(download_folder_paths[0])[6], variable_list=['ALTR', 'PTCH', "ALT"])