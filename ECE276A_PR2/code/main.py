import matplotlib.pyplot as plt
from scipy.linalg import expm
import numpy as np
import load_data
from load_data import *
import pr2_utils
from pr2_utils import *
import cv2 
import transforms3d

"""Motion Model"""
def motion_model(xt, v, imu_yaw_filtered_truncated, imu_stamps_truncated):
  noise = np.random.normal(loc=0.0, scale=0.005, size=(3, 100))
  xt = [(xt[0] + imu_tau*(v*np.cos(xt[2]))), (xt[1] + imu_tau*(v*np.sin(xt[2]))), (xt[2] + 2*omegatau)]  + noise
  
  # xt = [(xt[0] + imu_tau*(v*np.cos(xt[2]))), (xt[1] + imu_tau*(v*np.sin(xt[2]))), (xt[2] + 2*omegatau)]  
  xt = np.array(xt)
  
  return xt

"""Lidar Scanning"""
def lidar_scan():
  for i in range(len(imu_stamps_truncated)):
    for j in range(len(lidar_ranges)):
       x[i,j] = lidar_ranges[j, i]*np.cos(lidar_angle_min + j*lidar_angle_increment) + 0.13673
       y[i,j] = lidar_ranges[j, i]*np.sin(lidar_angle_min + j*lidar_angle_increment)
  return x, y


"""Mapping Function"""
def mapping(grid_world, world_coordinates, pose):
  resolution = 20
  xmax = 60
  ymax = 60
  for i in range(1081):
    endx = int(0.5*xmax*resolution) - int(world_coordinates[0,i]*resolution)
    endy = int(0.5*ymax*resolution) - int(world_coordinates[1,i]*resolution)
    lines = bresenham2D((int(0.5*resolution*xmax)-int(resolution*pose[0])), (int(0.5*resolution*ymax)-int(resolution*pose[1])), endx, endy)
    obstacle = [int(lines[0,-1]), int(lines[1,-1])]
    lines = lines[:,:-1]
    for line in range(len(lines[0])):
      grid_world[int(lines[0,line]), int(lines[1,line])] = 0
    grid_world[int(obstacle[0]),int(obstacle[1])] = 1
  return grid_world


"""Normalization"""
def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)


"""IMU filtering and truncating"""
imu_stamps_truncated = []
imu_stamps_truncated_indices = []
i, j = 0, 0
for i in range(len(encoder_stamps)):
  while j < len(imu_stamps):
    if(abs(encoder_stamps[i] - imu_stamps[j]) < 0.5):
      imu_stamps_truncated.append(imu_stamps[j])
      imu_stamps_truncated_indices.append(j)
      break
    else: j+=1
print((imu_stamps_truncated))

imu_yaw = imu_angular_velocity[2]
imu_yaw_filtered = np.zeros([len(imu_stamps)])
for i in range(1, len(imu_stamps)):
  imu_yaw_filtered[i] = 0.939*imu_yaw_filtered[i-1] + 0.0304*imu_yaw[i] + 0.0304*imu_yaw[i-1]

imu_yaw_filtered_truncated = []
for i in imu_stamps_truncated_indices:
  imu_yaw_filtered_truncated.append(imu_yaw_filtered[i])

plt.plot(imu_yaw)
plt.show()
plt.plot(imu_yaw_filtered)
plt.show()
plt.plot(imu_yaw_filtered_truncated)
plt.show()


"""Encoder to velocity"""
FR, FL, RR, RL = encoder_counts[0],encoder_counts[1], encoder_counts[2], encoder_counts[3]
v = np.zeros(len(FL))
for i in range(1, len(FR)):
  vR = 0.0011*(FR[i]+RR[i])/(encoder_stamps[i]-encoder_stamps[i-1])
  vL = 0.0011*(FL[i]+RL[i])/(encoder_stamps[i]-encoder_stamps[i-1])
  v[i] = 0.5*(vR+vL)

x = np.zeros([4962, 1081])
y = np.zeros([4962, 1081])
x, y = lidar_scan()

"""Particle filter SLAM (Prediction and Update)"""
all_world_coordinates = []
all_world_coordinates1 = []
grid_world = 0.5*np.ones([1200, 1200])
xt_list = []
no_particle = 100
particle = np.zeros([3, no_particle])
alpha = np.ones([1, no_particle])/no_particle
xim = np.arange(-30, -30+0.05, 0.05)
yim = np.arange(-30, -30+0.05, 0.05)
xs = np.arange(-0.2, 0.2+0.05, 0.05)
ys = np.arange(-0.2, 0.2+0.05, 0.05)
correlation = np.zeros([no_particle])
for i in range(len(imu_stamps_truncated)):
  imu_tau = imu_stamps_truncated[i] - imu_stamps_truncated[i-1]
  omegatau = 0.5*imu_tau*imu_yaw_filtered_truncated[i]
  for k in range(no_particle):
     transformationMatrix = np.array([[np.cos(particle[2,k]), -np.sin(particle[2,k]), particle[0,k]], 
                                      [np.sin(particle[2,k]),  np.cos(particle[2,k]), particle[1,k]],
                                      [            0,              0,     1]])
     world_coordinates = np.matmul(transformationMatrix, np.array([x[i,:], y[i,:] , np.ones(1081)]))
     grid_world = mapping(grid_world, world_coordinates, particle[:, k])
     
     particle[:,k] = motion_model(particle[:,k], v[i], omegatau, imu_tau)
     

     transformationMatrix1 = np.array([[np.cos(particle[2,k]), -np.sin(particle[2,k]), particle[0,k]], 
                                      [np.sin(particle[2,k]),  np.cos(particle[2,k]), particle[1,k]],
                                      [            0,              0,     1]])
     world_coordinates1 = np.matmul(transformationMatrix1, np.array([x[i+1,:], y[i+1,:] , np.ones(1081)]))
     s = np.stack((world_coordinates1[0], world_coordinates1[1]))
     correlation = mapCorrelation(grid_world, xim, yim, s, xs, ys)
     correlation = np.max(correlation)
     plt.imshow(grid_world, cmap='gray')
     plt.show()
     alpha_new[k] = alpha[k] * correlation[i, k]
  norm = np.linalg.norm(np.dot(alpha, correlation))
  alpha_new = alpha_new / np.sum(alpha_new)
  l = np.argmax(alpha_new)
  transform_matrix_update = np.array([[np.cos(particle[2,k]), -np.sin(particle[2,k]), particle[0,k]], 
                                      [np.sin(particle[2,k]),  np.cos(particle[2,k]), particle[1,k]],
                                      [            0,              0,     1]])
  

  particle_pos_world_up = np.zeros((lidar_ranges[:, 0].shape[0], 3))
  loc = np.zeros((4, 1))
  for j in range(lidar_ranges[:, 0].shape[0]):
      vec = np.array([particle[j, 0, i + 1], particle[j, 1, i + 1], particle[j, 2, i + 1], 1])
      loc = np.matmul(transform_matrix_update, vec)
      particle_pos_world_up[j, 0] = loc[0]
      particle_pos_world_up[j, 1] = loc[1]
      particle_pos_world_up[j, 2] = loc[2]
      # create the points in grid frame for body and lidar
      grid_frame_lidar = np.zeros((lidar_ranges[:, 0].shape[0], 2, len(encoder_stamps)))
      for j in range((lidar_ranges[:, 0].shape[0])):
          grid_frame_lidar[j, 0, i + 1] = int(1200 * 0.5) - int(20 * particle_pos_world_up[j, 0])
          grid_frame_lidar[j, 1, i + 1] = int(1200 * 0.5) - int(20 * particle_pos_world_up[j, 1])
      grid_frame_par_up = np.zeros((len(encoder_stamps), 2))
      grid_frame_par_up[i + 1, 0] = int(1200 * 0.5) - int(20 * particle[i + 1, 0, l])
      grid_frame_par_up[i + 1, 1] = int(1200 * 0.5) - int(20 * particle[i + 1, 1, l])
  for j in range((lidar_ranges[:, 0].shape[0])):
      a = bresenham2D(grid_frame_par_up[i + 1, 0], grid_frame_par_up[i + 1, 1], grid_frame_lidar[j, 0, i + 1],
                      grid_frame_lidar[j, 1, i + 1])
      for m in range(0, a.shape[1]):
          if np.absolute(int(a[0, m])) < 3000:
              if np.absolute(int(a[1, m])) < 3000:
                  occupancy_grid[int(a[0, m]), int(a[1, m])] = 50
                  occupancy_grid_log[int(a[0, m]), int(a[1, m])] = occupancy_grid_log[int(a[0, m]), int(a[1, m])] + np.log(2)
                  occupancy_grid[int(a[0, -1]), int(a[1, -1])] = 255
                  occupancy_grid_log[int(a[0, m]), int(a[1, m])] = occupancy_grid_log[int(a[0, m]), int(a[1, m])] + np.log(4)

  if 1 / (np.sum(np.square(alpha_new))) < no_particle / 10:
    particle[i + 1, 0, :] = np.random.choice(particle[i + 1, 0, :], 100, replace=True, p=alpha_new)
    particle[i + 1, 1, :] = np.random.choice(particle[i + 1, 1, :], 100, replace=True, p=alpha_new)
    particle[i + 1, 2, :] = np.random.choice(particle[i + 1, 2, :], 100, replace=True, p=alpha_new)
    
grid_world = 0.5*np.ones([1200, 1200])
for i in range(len(all_world_coordinates)):
  grid_world = mapping(grid_world, all_world_coordinates[i], xt_list[i])

disp_path = "/content/drive/MyDrive/276PR2/ECE276A_PR2/data/dataRGBD/Disparity20"
rgb_path = "/content/drive/MyDrive/276PR2/ECE276A_PR2/data/dataRGBD/RGB20"

T = np.zeros([4,4])
rot = transforms3d.euler2mat(0,0.36,0.021)

textureMap = np.zeros((2500, 2500,3), dtype=np.int64)

T = np.array([[rot[0,0],rot[0,1],rot[0,2], 0.181],
                 [rot[1,0],rot[1,1],rot[1,2], 0.005],
                 [rot[2,0],rot[2,1],rot[2,2], 0.660],
                 [0,0,0,1]])


"""Texture mapping"""
for i in range(imu_stamps_truncated):
# load RGBD image
    imd = cv2.imread(disp_path+'disparity20_1.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    imc = cv2.imread(rgb_path+'rgb20_1.png')[...,::-1] # (480 x 640 x 3

    # convert from disparity from uint16 to double
    disparity = imd.astype(np.float32)
    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd
    # calculate u and v coordinates 
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
  
    # get 3D coordinates 
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z
    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
    # display valid RGB pixels
    transformationMatrix = np.array([[np.cos(particle[2,k]), -np.sin(particle[2,k]), particle[0,k]], 
                                      [np.sin(particle[2,k]),  np.cos(particle[2,k]), particle[1,k]],
                                      [            0,              0,     1]])

    T = np.matmul(transformationMatrix, T)
    p_global = np.matmul(T, p.transpose(1, 0, 2))
    p_global = p_global.transpose(1, 0, 2)

    mask = (p_global[2, :, :] >= -0.18) & (p_global[2, :, :] <= 0.18) & (p[1, :, :] <= 0.6) & (p[1, :, :] >= -0.6)
    indices = np.where(mask&valid)
    indices = np.array(indices)

    xt = np.round(((p_global[0,indices[0,:],indices[1,:]])*45)+500).astype(np.int32)
    yt = np.round((-(p_global[1,indices[0,:],indices[1,:]])*45)+1250).astype(np.int32)
    textureMap[xt,yt,:] = imc[rgbv[mask&valid].astype(int),rgbu[mask&valid].astype(int)]
    # display disparity image
    plt.imshow(normalize(imd), cmap='gray')
    plt.show()