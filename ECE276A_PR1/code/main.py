"""Import Libraries"""
import load_data
from load_data import *
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import transforms3d

"""Quaternion tranformations"""
@jax.jit
def qlog(q):
    q = q + jnp.array([0.0] + [0.0001]*3)
    term1 = q[1:]*jnp.arccos(q[0]/qnorm(q))/qnorm(jnp.array([0, q[1], q[2], q[3]]))
    return jnp.array([jnp.log(qnorm(q)), term1[0], term1[1], term1[2]])

@jax.jit
def qnorm(q):
    return jnp.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])

@jax.jit
def qexp(q):
    qv_norm = qnorm(jnp.array([0, q[1], q[2], q[3]]))
    term1 = jnp.cos(qv_norm)
    term2 = jnp.array([0, q[1], q[2], q[3]])*(jnp.sin(qv_norm))/qv_norm
    return jnp.exp(q[0])*jnp.array([term1, term2[1], term2[2], term2[3]]) 

@jax.jit
def qmultiply(q0, q1):
    w0, x0, y0, z0 = q0[0], q0[1], q0[2], q0[3]
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    return jnp.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                     x1*w0 + y1*z0 - z1*y0 + w1*x0,
                    -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                     x1*y0 - y1*x0 + z1*w0 + w1*z0])

@jax.jit
def qinv(q):
    return jnp.array([q[0], -q[1], -q[2], -q[3]])/jnp.square(qnorm(q))

""" Coordinate frame trasformations"""
def spherical_to_cartesian(phi, theta):
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    # r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan(y/x)
    phi = np.arccos(z)
    return 1, theta, phi

"""Motion Model and Observation Model"""
def motionModel(omega, tau, q):
  q_exp = qexp([0, 0.5*tau*omega[0], 0.5*tau*omega[1], 0.5*tau*omega[2]])
  return jnp.array(qmultiply(q, q_exp))

def observeModel(gravity, q):
  return jnp.array(qmultiply(qmultiply(qinv(q), gravity), q))

def motionobservationModel(q):
  all_quat = list()
  all_accel = list()
  grav = jnp.array([0, 0, 0, -9.8])
  for i in range(len(imud["ts"][0])-1):
    tau = imud["ts"][0][i+1] - imud["ts"][0][i]
    all_quat.append(q)
    a = observeModel(grav, q)
    q = motionModel(unbiased_omega[i], tau, q)
    all_accel.append(a)

  return all_quat, all_accel

"""Cost Function"""
def cost_function(qt):
    cost = 0
    grav = jnp.array([0, 0, 0, g])
    for i in range(len(qt)-1):
      tau = imud["ts"][0][i+1] - imud["ts"][0][i]
      q_motion = motionModel(unbiased_omega[i], tau, qt[i])
      acc_observe = observeModel(grav, qt[i])
      # term1 = jnp.square(qnorm(2*qlog(qmultiply(qinv(qt[i+1]), q_motion))))
      term1 = 0
      term2 = jnp.square(qnorm(jnp.subtract(jnp.array([0, unbiased_acc[i][0], unbiased_acc[i][1], unbiased_acc[i][2]]),jnp.array([0, acc_observe[1], acc_observe[2], acc_observe[3]]))))
      cost += 0.5*(term1 + term2)
    return cost

"""Angle transformations"""
all_euler_angles = []
def quaternionToEuler(all_quaternion):
  for i in range(len(all_quaternion)):
    euler_anles = transforms3d.euler.quat2euler(all_quaternion[i])
    all_euler_angles.append(euler_anles)
  return all_euler_angles

all_euler_angles_final = []
def quaternionToEulerFinal(all_quaternion):
  for i in range(len(all_quaternion)):
    euler_anles = transforms3d.euler.quat2euler(all_quaternion[i])
    all_euler_angles_final.append(euler_anles)
  return all_euler_angles_final

all_vicon_angles = []
def viconToEuler(all_vicon_rot):
  for i in range(len(all_vicon_rot)):
    vicon_anles = transforms3d.euler.mat2euler(all_vicon_rot[i])
    all_vicon_angles.append(vicon_anles)
  return all_vicon_angles

"""Parameter initialisation"""

vref, g, sensitivity_acc, sensitivity_omega, delta, count = 3300, -9.8, 1/300, np.pi/(3.33*180), 0.0005, 0
scale_factor_acc, scale_factor_omega = vref*sensitivity_acc/1023, vref*sensitivity_omega/1023
value_acc, value_omega = jnp.array([0, 0, g]), jnp.array([0, 0, 0])
all_unbiased_acc, all_unbiased_omega, all_quaternion, all_acc, bias_acc, bias_omega,rot = [], [], [], [], [], [], []
bias_omega_sum, bias_acc_sum = jnp.array([0., 0., 0.]), jnp.array([0., 0., 0.])
all_vicon_rot = []
count = 0

"""Main code: Part 1"""
for i in range(len(vicd["ts"][0])):
    rot = np.array(vicd["rots"][:,:,i])
    all_vicon_rot.append(rot)
    del_mat = jnp.subtract(jnp.eye(3), rot)
    if del_mat[0][0] < delta and del_mat[1][1] < delta and del_mat[1][1] < delta:
        count += 1
    else: continue
all_vicon_rot = np.array(all_vicon_rot)
all_indices = []

Ax, Ay, Az = imud["vals"][0], imud["vals"][1], imud["vals"][2]
a = []

for i in range(len(imud["vals"][0])):
  acl = np.array([Ax[i], Ay[i], Az[i]])
  a.append(acl)
a=np.array(a)

bias_acc_array = []
bias_omega_array = []
for i in range(count):
  bias_acc = np.array([imud["vals"][0,i], imud["vals"][1,i], imud["vals"][2,i]])
  bias_omega = np.array([imud["vals"][4,i], imud["vals"][5,i], imud["vals"][3,i]])
  bias_acc_array.append(bias_acc)
  bias_omega_array.append(bias_omega)

bias_acc_av = np.mean(bias_acc_array, axis=0)
bias_omega_av = np.mean(bias_omega_array, axis=0)
unbiased_omega = list()
unbiased_acc = list()

for i in range(len(imud["vals"][0])):
  acc = (a[i] - bias_acc_av)*scale_factor_acc*(-9.8)
  acc[0] =acc[0]
  acc[1] = acc[1]
  acc[2] = acc[2] -9.8
  unbiased_acc.append(acc)
  omega = (np.array([imud["vals"][4,i], imud["vals"][5,i], imud["vals"][3,i]]) - bias_omega_av)*scale_factor_omega
  unbiased_omega.append(omega)
unbiased_omega = np.array(unbiased_omega)
unbiased_acc = np.array(unbiased_acc)


q = jnp.array([1.0,0.0,0.0,0.0])
all_quaternion, all_acc = motionobservationModel(q)
all_acc = np.array(all_acc)

all_quaternion = np.array(all_quaternion)
motionquat = quaternionToEuler(all_quaternion)
motionquat = np.array(motionquat)
vicon_angles = viconToEuler(all_vicon_rot)
vicon_angles = np.array(vicon_angles)

"""Gradient Descent Algorithm"""
iter = 20
qt = jnp.array(all_quaternion)
for i in range(iter):
  q = [qt[j]/qnorm(qt[j]) for j in range(len(qt))]
  cost = cost_function(q)
  print("iteration:", i, "Cost:", cost)
  gradient = jax.grad(cost_function)
  delta = jnp.array(gradient(q))
  q = jnp.array(q)
  qt = jnp.subtract(q, (0.001*delta))

qt = np.array(qt)
finalquat = quaternionToEulerFinal(qt)
finalquat = np.array(finalquat)

vicon_angles = np.array(vicon_angles)
all_acc = np.array(all_acc)

"""Panorama construction"""
for i in range(len(camd['ts'][0])):
  for j in range(len(vicd['ts'][0])):
    if 0. <= camd['ts'][0, i] - vicd['ts'][0,j] <= .01074:
      all_indices.append(j) 
    else: continue

all_indices = np.array(all_indices)
vicon_rot_camera = []
for i in all_indices:
  vicon_rot_camera.append(all_vicon_rot[i])

vicon_rot_camera = np.array(vicon_rot_camera)

mat_spherical = np.empty([240, 320, 3])
mat_cartesian = np.empty([240, 320, 3])
world_coordinates = np.empty([3, 76800])
world_to_spherical = np.empty([3, 76800])
all_world_to_spherical = np.empty([len(vicon_rot_camera), 3, 76800])
factor = 0.0032724923474893677

for i in range(0, 240):
  for j in range(0, 320):
    mat_spherical[i, j] = [1, i*factor, j*factor]
    mat_cartesian[i, j] = spherical_to_cartesian(mat_spherical[i, j, 1], mat_spherical[i, j, 2])

mat_cartesian_vector = np.transpose(mat_cartesian).reshape(3, 320*240)

for k in range(len(vicon_rot_camera)):
  world_coordinates = np.matmul(vicon_rot_camera[k], mat_cartesian_vector)
  world_to_spherical = np.array([np.ones(76800), np.arccos(world_coordinates[2]), np.arctan2(world_coordinates[1],world_coordinates[0])])
  all_world_to_spherical[k] = world_to_spherical

all_world_to_spherical[:, 2, :]+=np.pi
all_world_to_spherical[:, 1, :]/=(np.pi/1080)
all_world_to_spherical[:, 2, :]/=(2*np.pi/1920)

output_image = np.zeros([1080, 1920, 3])

"""RGB value assignment"""
all_world_to_spherical = all_world_to_spherical.astype(np.int32)
for k in range(0, len(camd["cam"][239, 319, 2, :]), 10):
  for i in range(76800):
    y, x = all_world_to_spherical[k, 1, i], all_world_to_spherical[k, 2, i]
    output_image[y, x, :] = camd["cam"][i//320, i%240, :, k]

"""Final Panorama output image"""
output_im = output_image/255
plt.imshow(output_im)


#################################################################
# plt.plot(unbiased_omega[:,0])
# plt.plot(unbiased_omega[:,1])
# plt.plot(unbiased_omega[:,2])
# plt.plot(unbiased_acc[:,0])
# plt.plot(unbiased_acc[:,1])
# plt.plot(unbiased_acc[:,2])
#################################################################

# plt.figure(1)
# plt.plot(finalquat[:,0],label="Motion Model Angle X",c="b")
# plt.plot(motionquat[:,0],label="Motion Model Angle Y",c="r")
# plt.plot(vicon_angles[:,0],label="Vicon Angle X",c="orange")
# plt.xlim([0,5645])
# plt.legend()

# plt.figure(2)
# plt.plot(finalquat[:,1],label="Motion Model Angle Y",c="b")
# plt.plot(motionquat[:,1],label="Motion Model Angle Y",c="r")
# plt.plot(vicon_angles[:,1],label="Vicon Angle Y",c="orange")
# plt.xlim([0,5645])
# plt.legend()

# plt.figure(3)
# plt.plot(finalquat[:,2],label="Motion Model Angle Z",c="b")
# plt.plot(motionquat[:,2],label="Motion Model Angle Y",c="r")
# plt.plot(vicon_angles[:,2],label="Vicon Angle Z",c="orange")
# plt.xlim([0,5645])
# plt.legend()
##################################################################

# plt.figure(1)
# plt.plot(unbiased_acc[:,0],c='r')
# plt.plot(all_acc[:,1])

# plt.figure(2)
# plt.plot(unbiased_acc[:,1],c='r')
# plt.plot(all_acc[:,2])

# plt.figure(3)
# plt.plot(unbiased_acc[:,2],c='r')
# plt.plot(all_acc[:,3])
# plt.show()