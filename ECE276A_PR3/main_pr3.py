import numpy as np
import scipy
import matplotlib.pyplot as plt
import transforms3d
from pr3_utils import *

if __name__ == '__main__':

	filename = "/content/drive/MyDrive/276PR3/data/03.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

def isVisible(landmarks):
    comparison = -1 * np.ones([4, len(landmarks[0, :])])
    difference = comparison - landmarks
    visibility = np.all(difference, axis=0)
    return np.where(visibility)[0]

def pose2axangle(pose):
  R = pose[:3,:3]
  position = pose[:3,3]
  axis, angles = transforms3d.axangles.mat2axangle(R)
  axangles = axis*angles
  return np.concatenate((position, axangles))

def axangletopose(axangle):
  position = axangle[:3]
  angle = np.linalg.norm(axangle[-3:])
  axis = axangle[-3:] / angle
  R = transforms3d.axangles.axangle2mat(axis, angle)
  pose = np.zeros((4,4))
  pose[:3,:3] = R
  pose[:3,3] = position
  pose[3,3] = 1
  return pose

def inverseCam(feature_at_t):
    fsu = K[0,0]
    fsv = K[1,1]
    cu = K[0,2]
    cv = K[1,2]
    b = 0.6
    x = (feature_at_t[0] - cu)/fsu
    y = (feature_at_t[1] - cv)/fsv
    z = (fsu*b)/(feature_at_t[0]- feature_at_t[2])
    return (np.array([x,y,z,1]))

def kalmanGain(kalmanGain, n):
  kal_reshape = kalmanGain.reshape(-1, 3, 4)
  return np.hstack(kal_reshape.repeat(n, axis=0))

# visualize_trajectory_2d(np.transpose(all_T,(1,2,0)), show_ori = True)

Ks = np.block([[K[0, 0],       0, K[0, 2],            0],
               [      0, K[1, 1], K[1, 2],            0],
               [K[0, 0],       0, K[0, 2], -K[0, 0] * b],
               [      0, K[1, 1], K[1, 2],            0]])
# J = Ks
pi = projection(features[:,:,0])
z = Ks
visible_indices = []
for i in range(len(features[0,0,:])):
  print(i)
  visible = features[0,:,i] - features[1,:,i] + features[2,:,i] - features[3,:,i]
  visible_indices.append(np.where(visible != 0))

a = [np.eye(3)]*5000
b = np.array([a]*5000)

cov_matrix = np.array([[np.eye(3)]*features.shape[1]]*features.shape[1])

fsu = K[0,0]
fsv = K[1,1]
cu = K[0,2]
cv = K[1,2]
b = 0.6
P = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]


Ks = [[fsu, 0, cu, 0],[0,fsv,cv,0],[fsu,0,cu,-fsu*b],[0,fsv,cv,0]]

mult_visit = np.zeros((features.shape[1]))
camtocart_lib = np.zeros((4,features.shape[1]))
z_tilda = np.zeros((4,features.shape[1]))

mean_landmarks = np.ones([4,len(features[0,:,0])])
cov_landmarks = np.array([[0.001*np.eye(3)] * len(features[0,:,0])] * len(features[0,:,0]))
visited = dict()

mean = np.eye(4)
cov = 0.01*np.eye(6)
last = np.concatenate((np.zeros([3,3]), 0.1*np.eye(3)), axis=1)
W = np.concatenate((np.concatenate((np.eye(3), np.zeros([3,3])),axis=1), last), axis=0)
mean_robot = np.zeros([len(t), 4, 4])
cov_robot = np.zeros([len(t), 6, 6])
mean_robot[0,:] = mean
cov_robot[0,:] = cov

z_tilda = np.zeros([4,len(features[0,:,0])])
H_matrix = np.array([np.zeros([4, 3])] * len(features[0,:,0]))
visited = []

"""EKF Prediction Localization part 1"""
for time_step in range(1, len(t)):

  tau = t[time_step] - t[time_step-1]
  T = np.eye(4)
  all_T = []
  all_T.append(T)
  for i in range(len(np.transpose(t))-1):
    tau = t[0, i+1] - t[0, i]
    eta = scipy.linalg.expm((tau*np.array([[                      0, -angular_velocity[2, i],  angular_velocity[1, i], linear_velocity[0, i]],
                                         [ angular_velocity[2, i],                       0, -angular_velocity[0, i], linear_velocity[1, i]],
                                         [-angular_velocity[1, i],  angular_velocity[0, i],                       0, linear_velocity[2, i]],
                                         [                      0,                       0,                       0,                    0]])))
  T = np.matmul(T, eta)
  all_T.append(T)

  all_T = np.array(all_T)
  all_T.shape
  x = all_T[:, 0, 3]
  y = all_T[:, 1, 3]
  plt.plot(x,y)
  velocity = np.array([linear_velocity[0, time_step], linear_velocity[1, time_step], linear_velocity[2, time_step], angular_velocity[0, time_step], angular_velocity[1, time_step], angular_velocity[2, time_step]])
  twist = axangle2twist(velocity)
  adtwist = axangle2adtwist(velocity)

  predicted_mean = np.matmul(mean_robot[time_step-1,:,:],twist2pose(delta_timetwist))
  predicted_covariance = np.matmul(np.matmul(pose2adpose(twist2pose(-delta_timeadtwist)),cov_robot[time_step-1,:,:]),np.transpose(pose2adpose(twist2pose(-delta_time*adtwist)))) + W

  mean_robot[time_step, :,:] = predicted_mean
  cov_robot[time_step, :, :] = predicted_covariance

  visible = isVisible(features[:, :, 1:len(t)])
  visited = np.zeros(features.shape[1], dtype=int)
  mean_landmarks_t = np.zeros((4, features.shape[1]))
  cov_landmarks_t = np.zeros((features.shape[1], 3, 3))


"""Mapping EKF Update part 2"""
for t in range(1, len(t)):
    visible_t = visible[:, :, t-1]
    revisited_t = np.isin(visible_t, np.where(visited == 1))[0]
    visible_t = np.setdiff1d(visible_t, revisited_t)
    
    mean_landmarks_t[:,visible_t] = mean_robot[t,:,:]@imu_T_cam@inverseCam(features[:,visible_t,t])
    visited[visible_t] = 1
    
    z_tilda= np.matmul(Ks, projection(np.matmul(inversePose(imu_T_cam), inversePose(mean_robot[t, :, :])), mean_landmarks_t[:3,visible_t]))  
    P = np.concatenate((np.eye(3), np.array([[0],[0],[0]])), axis=1)
    Jaco = np.matmul(projectionJacobian(inversePose(imu_T_cam), inversePose(mean_robot[t,:,:]), mean_landmarks_t[:3,visible_t]))
    temp = np.matmul(inversePose(imu_T_cam), inversePose(mean_robot[t,:,:]), np.transpose(P))  
    H_matrix = np.matmul(K_intrinsic, Jaco, temp)
    
    kalman_gain = np.zeros((4, 3, visible_t.shape[0]))
    for i, j in enumerate(visible_t):
        kalman_gain[:,:,i] = np.matmul(cov_landmarks_t[j,:,:], H_matrix[:,i,:].T, np.linalg.inv(np.matmul(H_matrix[:,i,:]cov_landmarks_t[j,:,:]H_matrix[:,i,:].T + np.eye(4))))
    
    diff_z = (features[:,visible_t,t] - z_tilda)
    mean_landmarks_t[:3,visible_t] = mean_landmarks_t[:3,visible_t] + np.matmul(kalman_gain.transpose(2,0,1), diff_z)
    
    for i, j in enumerate(visible_t):
        cov_landmarks_t[j,:,:] = (np.eye(3) - kalman_gain[:,:,i]@H_matrix[:,i,:])@cov_landmarks_t[j,:,:]
cov_landmarks = np.concatenate(np.concatenate(cov_landmarks_t, axis=1), axis=1)

"""Visual SLAM part 3"""
num_features = len(features[0,:,0])
landmark_cov_size = num_features * 3
landmark_cov = np.eye(landmark_cov_size) * 1

landmark_cov = np.concatenate([landmark_cov] * num_features)

robot_cov = np.eye(6) * 0.001
cov_size = landmark_cov_size + 6
covariance = np.zeros([cov_size, cov_size])
covariance[:landmark_cov_size, :landmark_cov_size] = landmark_cov
covariance[-6:,-6:] = robot_cov
I_cov = np.eye(covariance.shape[0])

num_time_steps = len(time)
robot_mean = np.zeros([num_time_steps, 4, 4])
robot_mean[0] = np.eye(4)
landmark_mean = np.zeros([4, num_features])

visited = set()

W = np.zeros([6,6])
W[:3,:3] = np.eye(3) * 0.01
W[-3:,-3:] = np.eye(3) * 0.001

for t in range(1, len(t)):
    tau = t[t] - t[t - 1]

    u = np.array([linear_velocity[0, t - 1], linear_velocity[1, t - 1], linear_velocity[2, t - 1], 
                  angular_velocity[0, t - 1], angular_velocity[1, t - 1], angular_velocity[2, t - 1]])
    u_hat = axangle2twist(u)
    u_curly_hat = axangle2adtwist(u)

    robot_mean[t] = np.matmul(robot_mean[t - 1], twist2pose(tau * u_hat))  
    covariance[-6:, -6:] = np.matmul(twist2pose(-tau * u_curly_hat) , covariance[-6:, -6:], (twist2pose(-tau * u_curly_hat).T)) + W  
    visible = isVisible(features[:, :, t])
    revisited = []
    z_tilda_f = []
    H_slam = []
    
    for i in visible[0]:
        if features[3, i, t] - features[2, i, t] != 0:
            if i not in visited:
                landmark_mean[:, i] = np.matmul(robot_mean[t], imu_T_cam, inverseCam(features[:, i, t]))
                visited.add(i)
            else:
                landmark_H = np.zeros((4, 3 * len(features[0, :, 0])))
                revisited.append(i)
                landmark_pose = np.matmul(inversePose(imu_T_cam), inversePose(robot_mean[t]), landmark_mean[:, i])
                z_tilda_f.append(np.matmul(K_intrinsic, projection(landmark_pose))) 
                P = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
                landmark_H[:, 3 * i:3 * i + 3] = np.matmul(K_intrinsic, projectionJacobian(landmark_pose), 
                                                 inversePose(imu_T_cam), inversePose(robot_mean[t]), P.T)  
    if len(revisited) > 0:
      H = np.concatenate(H_slam, axis=0)
      kalman_gain = np.matmul(covariance, H_slam.T, np.linalg.inv(np.matmul(H_slam,covariance,H_slam.T + 5*np.eye(4*len(revisited)))))
      z_tilda = np.concatenate(z_tilda_f, axis=0)
      diff_z = np.ravel(features[:,revisited,t], order='F') - z_tilda

  
      kalman_gain_robot = kalman_gain[-6:,:]
      kalman_gain_landmark = kalman_gain[:-6,:]
      landmark_mean_reshaped = np.concatenate(landmark_mean[:3,:], axis=0)
      landmark_mean[:3,:] = (landmark_mean_reshaped + np.matmul(kalman_gain_landmark,diff_z)).reshape(3, len(features[0,:,0]))
      robot_mean[t,:,:] = np.matmul(robot_mean[t,:,:],twist2pose(axangle2twist(np.squeeze((np.matmul(kalman_gain_robot,diff_z)))))
      covariance = np.matmul((I_cov - np.matmul(kalman_gain, H_slam)),covariance))

