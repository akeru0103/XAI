'''dqn settings'''
EPISODE_NUMBER = 500

dqn_a = 4
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LEARNING_RATE=0.001

UPDATE_TARGET_Q_FREQ = 5
TRAIN_FREQ = 3
MODEL_SAVE_FREQ = 10

'''saliency settings'''
SALIENCY_SAVING = False
SALIENCY_ROUGHNESS = 8

'''ndarray save dettings'''
SAVE_SCREEN = True
SAVE_FREQUENCY = 100
START_DURATION = 5
START_SAVE_FREQUENCY = 3
END_DURATION = 5
END_SAVE_FREQUENCY = 3


# -*- coding: utf-8 -*-

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image, ImageDraw, ImageEnhance

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from scipy.ndimage.filters import gaussian_filter
import os
import pickle
from statistics import mean
import datetime
import time

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16*dqn_a, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16*dqn_a)
        self.conv2 = nn.Conv2d(16*dqn_a, 32*dqn_a, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32*dqn_a)
        self.conv3 = nn.Conv2d(32*dqn_a, 32*dqn_a, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32*dqn_a)
        self.head = nn.Linear(448*dqn_a, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# This is based on the code from gym.
screen_width = 600

def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen


    '''
    if SAVE_SCREEN==True and save_screen==True and i_episode % SAVE_FREQUENCY==0:
        screen_sequence.append(screen/255)
    '''

    screen = screen[:, 160:320]

    view_width = 320

    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

env.reset()

policy_net = DQN().to(device)
q_ast = deepcopy(policy_net)
print(policy_net)

target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(),lr=LEARNING_RATE)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():

            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= START_DURATION+1:#100:
        #means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        #means = torch.cat((torch.zeros(99), means))
        means = durations_t.unfold(0, START_DURATION+1, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(START_DURATION), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


ave_loss = []
ave_q = []
def plot_loss():
    plt.figure(3)
    plt.clf()
    loss_t = torch.tensor(ave_loss, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss/Ave q')
    plt.plot(loss_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    #global last_sync
    q_ast = policy_net
    if len(memory) < BATCH_SIZE:
        episode_loss.append(0)
        return
    #print(num_episodes)
    if steps_done % TRAIN_FREQ == 0:
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #policy_net.eval()
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = q_ast(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        #policy_net.train()
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        #print(loss.item())
        #
        episode_loss.append(loss.item())
        #

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    if steps_done % UPDATE_TARGET_Q_FREQ == 0:
            q_ast = deepcopy(policy_net)

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def save_image(image, save_name, save_type='png'):
    if image.max() <= 1:
        image = image*255
    if os.path.exists(save_folder+'/images')==False:
        os.mkdir(save_folder+'/images')
    if image.ndim==3:
        image = Image.fromarray(np.uint8(image.transpose(1, 2, 0)))
        image.save(save_folder+'/images/'+save_name+'.'+save_type)
    elif image.ndim==2:
        image_width = image.shape[1]
        image_hight = image.shape[0]
        output_image = np.zeros((image_hight, image_width, 3))
        for i in range(3):
            output_image[:,:, i] = image
        image = Image.fromarray(np.uint8(output_image))
        image.save(save_folder+'/images/'+save_name+'.'+save_type)



def save_movie(image_sequence, save_name, save_size=(640,320), save_type='gif', frame_length=160, loop=0):

    if os.path.exists('images')==False:
        os.mkdir('images')


    movie = []
    for i in range( len(image_sequence) ):
        movie.append( Image.fromarray(np.uint8(image_sequence[i]*255).transpose(1, 2, 0)))
        movie[i] = movie[i].resize(save_size)


    movie[0].save('images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
def blend_save_movie(image_sequence1, image_sequence2, save_name, image_sequence1_rate=0.5, contrast_rate=1, save_type='gif', frame_length=160, loop=0):


    if len(image_sequence1)!=len(image_sequence2):
        print('error:blend_save_movie')
        print('length of image_sequence1 and image_sequence2 must be same')

    if os.path.exists('images')==False:
        os.mkdir('images')


    movie = []
    for i in range( len(image_sequence1) ):
        image1 = Image.fromarray(np.uint8(image_sequence1[i]*255).transpose(1, 2, 0))
        image2 = Image.fromarray(np.uint8(image_sequence2[i]*255).transpose(1, 2, 0))
        image1 = image1.resize((640, 320))
        image2 = image2.resize((640, 320))
        image2 = Image.blend(image2, image1, image_sequence1_rate)
        image2 = ImageEnhance.Contrast(image2).enhance( contrast_rate/max(image_sequence1_rate,1-image_sequence1_rate) ) #2)
        movie.append( image2 )


    movie[0].save('images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)


save_num = 0
def save_ndarray_list(ndarray_list, save_name, save_type='npz'):
    global save_num
    #print(' outputting '+save_name+f'{save_num:0=3}'+'.'+save_type+'...')
    if os.path.exists(save_folder+'/files')==False:
        os.mkdir(save_folder+'/files')

    #np.savez_compressed(save_folder+'/files/'+save_name+f'{save_num:0=3}'+'.'+save_type, ndarray_list)
    print(' finished')

def save_list(list, save_name, save_type='npz'):
    #global save_num
    print(' outputting '+save_name+'.'+save_type+'...')
    if os.path.exists(save_folder+'/files')==False:
        os.mkdir(save_folder+'/files')
    np.savez_compressed(save_folder+'/files/'+save_name+'.'+save_type, list)
    print(' finished')

def save_ndarrays():
    if SALIENCY_SAVING == True:
        global save_num
        save_ndarray_list(input_sequence, 'input')
        save_ndarray_list(saliency_map_sequence, 'saliency_map')
        save_list(saved_episode, 'episodes')
        save_list(saved_episode_rewards, 'rewards')
        if SAVE_SCREEN==True:
            save_ndarray_list(screen_sequence, 'screen')
            save_ndarray_list(cart_location_sequence, 'cart_location')
        save_num += 1

def make_perturbed_image(image, perturbed_point, mask_sigma, blurred_sigma, save=False):


    image_width = image.shape[2]
    image_hight = image.shape[1]
    mask = get_mask(perturbed_point, [image_hight,image_width], mask_sigma)

    blurred_frame = np.zeros((3, image_hight, image_width))
    image1 = np.zeros((3, image_hight, image_width))
    image2 = np.zeros((3, image_hight, image_width))
    image3 = np.zeros((3, image_hight, image_width))
    #image3 = copy.copy(image)#np.zeros((3, image_hight, image_width))

    for i in range(3):
        blurred_frame[i] = gaussian_filter(image[i], sigma=blurred_sigma)
        image1[i] = np.multiply(image[i],1-mask)
        image2[i] = blurred_frame[i]*mask
        image3[i] = image1[i] + image2[i]

    if save==True:
        save_image(image, 'input_image')
        save_image(mask, 'mask_image'+str(perturbed_point)+',s='+str(mask_sigma))
        save_image(1-mask, '1-mask_image'+str(perturbed_point)+',s='+str(mask_sigma))
        save_image(image1, 'input(1-mask_image)'+str(perturbed_point)+',s='+str(mask_sigma))
        save_image(image2, 'blurred_image_mask_image,sa='+str(blurred_sigma))
        save_image(image3, 'perturbed_image'+str(perturbed_point)+',s='+str(mask_sigma)+',sa='+str(blurred_sigma))
        save_image(blurred_frame, 'blurred_frame,sa='+str(blurred_sigma))

    return image3

def make_saliency_map(image, mask_sigma, blurred_sigma, decimation_rate):


    state_width = state.shape[3]
    state_hight = state.shape[2]
    normal_q = policy_net(image)
    saliency_map = np.zeros((int(state_hight/decimation_rate), int(state_width/decimation_rate) ))
    for i in range(0, state_width, decimation_rate):
        for j in range(0, state_hight, decimation_rate):
            perturbed_state = make_perturbed_image( (image.cpu().numpy().squeeze())*255, [j,i], mask_sigma, blurred_sigma )
            perturbed_state = torch.from_numpy(perturbed_state)
            perturbed_state = perturbed_state.unsqueeze(0).to(device).float()
            perturbed_q = policy_net(perturbed_state)
            saliency_map[int(j/decimation_rate), int(i/decimation_rate)] = float( (normal_q-perturbed_q).pow(2).sum().mul_(0.5) )
    return saliency_map

average_of_reward_max = 0
def decision_of_save(episode_num, average_of_reward, episode_per_saliency_start, episode_per_saliency_duration_start, episode_per_saliency_end, episode_per_saliency_duration_end):
    if EPISODE_NUMBER < episode_per_saliency_duration_start + episode_per_saliency_duration_end:
        print('error:decision_of_save')
        return -1
    global average_of_reward_max
    saliency_save_flag = False
    if SALIENCY_SAVING == False:
        return saliency_save_flag
    if episode_num < episode_per_saliency_duration_start:
        if episode_num % episode_per_saliency_start == 0:
            saliency_save_flag = True
    elif EPISODE_NUMBER - episode_per_saliency_duration_end <= episode_num:
        if episode_num % episode_per_saliency_end == 0:
            saliency_save_flag = True
    elif average_of_reward_max + 5 < average_of_reward[-1]:
        average_of_reward_max = average_of_reward[-1]
        saliency_save_flag = True
    elif episode_num == episode_per_saliency_duration_start or episode_num % SAVE_FREQUENCY == 0:
        saliency_save_flag = True
    return saliency_save_flag

def print_time(start_time):
    now = datetime.datetime.now()
    print(' elapsed time : '+str(now-start_time))

def find_folder_number():
    for i in count():
        if os.path.exists('results/result'+str(i+1))==False:
            return i+1
        if i>100:
            return -1

def make_lowest_folder():
    if os.path.exists('results')==False:
        os.mkdir('results')
    path = 'results/result'+str(find_folder_number())
    os.mkdir(path)
    os.mkdir( str(path)+'/files' )
    os.mkdir( str(path)+'/images' )
    return path

def get_input_position():
    view_width = 320

    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        position = view_width // 2
    elif cart_location > (screen_width - view_width // 2):
        position = screen_width - view_width // 2
    else:
        position = cart_location
    return position

def save_model(model, save_name):
    print(' outputting '+save_name+'.pth...')
    if os.path.exists(save_folder+'/files')==False:
        os.mkdir(save_folder+'/files')
    if os.path.exists(save_folder+'/files/'+save_name)==False:
        os.mkdir(save_folder+'/files/'+save_name)

    torch.save(model.state_dict(), save_folder+'/files/'+save_name+'/'+save_name+'.pth')
    print(' finished')



num_episodes = EPISODE_NUMBER
saliency_calcuration_rate = SALIENCY_ROUGHNESS

saved_episode = []
saved_episode_rewards = []
ave=0
ave_max=0
average_of_reward=[]
saliency_save_flag = False

start_time = datetime.datetime.now()
print(' start time : '+str(start_time))

save_folder = make_lowest_folder()

with open(save_folder+'/result.txt', mode='w')as f:
    f.write('\nsettings\n')
    f.write('EPISODE_NUMBER: '+str(EPISODE_NUMBER)+'\n')
    f.write('BATCH_SIZE: '+str(BATCH_SIZE)+'\n')
    f.write('GAMMA: '+str(GAMMA)+'\n')
    f.write('EPS_START: '+str(EPS_START)+'\n')
    f.write('EPS_END: '+str(EPS_END)+'\n')
    f.write('EPS_DECAY: '+str(EPS_DECAY)+'\n')
    f.write('TARGET_UPDATE: '+str(TARGET_UPDATE)+'\n')
    f.write('SALIENCY_SAVING: '+str(SALIENCY_SAVING)+'\n')
    f.write('SALIENCY_ROUGHNESS: '+str(SALIENCY_ROUGHNESS)+'\n')
    f.write('LEARNING_RATE: '+str(LEARNING_RATE)+'\n')
    f.write(str(policy_net)+'\n')


print(' saving variables...')
variables={"EPISODE_NUMBER":EPISODE_NUMBER, "SAVE_FREQUENCY":SAVE_FREQUENCY, "SALIENCY_ROUGHNESS":SALIENCY_ROUGHNESS, "SAVE_SCREEN":SAVE_SCREEN}
with open(save_folder+'/files/variables.pickle', mode='wb') as f:
    pickle.dump(variables, f)
print(' finished')

for i_episode in range(num_episodes):
    saliency_map_sequence = []
    input_sequence = []
    screen_sequence = []
    cart_location_sequence = []
    episode_loss = []
    episode_aveq = []
    # Initialize the environment and state
    env.reset()

    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    if ave > ave_max and i_episode >= START_DURATION+1:
        ave_max = ave
        episode_of_max_ave = i_episode

    ave_q.append(np.mean(policy_net(state).cpu().detach().numpy()))


    for t in count():


        if i_episode == 0 and t == 3:
            make_perturbed_image(state.squeeze().cpu().numpy(),(20,40),4,3,save=True)
            print(' filter images were saved')


        if saliency_save_flag == True:
            saliency_map_sequence.append( make_saliency_map(state, 4, 3, saliency_calcuration_rate ) )
            input_sequence.append( current_screen.cpu().numpy().squeeze() )
            if SAVE_SCREEN==True:
                screen_sequence.append(env.render(mode='rgb_array').transpose((2, 0, 1))/255)
                cart_location_sequence.append(get_input_position())

        episode_aveq.append(np.mean(policy_net(state).cpu().detach().numpy()))
        # Select and perform an action
        action = select_action(state)

        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()

        if done:
            episode_durations.append(t + 1)
            ave = mean(episode_durations)
            average_of_reward.append(mean(episode_durations))

            if saliency_save_flag==True:
                #print(' episode: '+str(i_episode)+' / '+str(EPISODE_NUMBER-1)+', reward: '+str(t+1)+', average/ave_max: '+f'{ave:.2f}'+'/'+f'{ave_max:.2f}'+' saliency was generated')
                saved_episode.append(i_episode)
                saved_episode_rewards.append(average_of_reward[-1])
                save_ndarrays()
            #else:
                #print(' episode: '+str(i_episode)+' / '+str(EPISODE_NUMBER-1)+', reward: '+str(t+1)+', average/ave_max: '+f'{ave:.2f}'+'/'+f'{ave_max:.2f}')
            saliency_save_flag = decision_of_save(i_episode, average_of_reward, START_SAVE_FREQUENCY, START_DURATION, END_SAVE_FREQUENCY, END_DURATION)

            ave_loss.append(mean(episode_loss)/np.mean(np.array(episode_aveq)))

            if i_episode % MODEL_SAVE_FREQ == 0:
                save_model(policy_net,'episode'+str(i_episode))

            if (i_episode+1) % 10 == 0:
                plt.figure(2)
                plt.savefig(save_folder+'/figure.png')
                plt.figure(3)
                plt.savefig(save_folder+'/loss.png')
                print_time(start_time)
                plot_durations()
                plot_loss()
            break

print(' number of saved episode : '+str(len(saved_episode)))
print(' saved episode number : '+str(saved_episode))


with open(save_folder+'/result.txt', mode='w')as f:
    f.write('\nsettings\n')
    f.write('EPISODE_NUMBER: '+str(EPISODE_NUMBER)+'\n')
    f.write('BATCH_SIZE: '+str(BATCH_SIZE)+'\n')
    f.write('GAMMA: '+str(GAMMA)+'\n')
    f.write('EPS_START: '+str(EPS_START)+'\n')
    f.write('EPS_END: '+str(EPS_END)+'\n')
    f.write('EPS_DECAY: '+str(EPS_DECAY)+'\n')
    f.write('TARGET_UPDATE: '+str(TARGET_UPDATE)+'\n')
    f.write('SALIENCY_SAVING: '+str(SALIENCY_SAVING)+'\n')
    f.write('SALIENCY_ROUGHNESS: '+str(SALIENCY_ROUGHNESS)+'\n')
    f.write('\nresults\n')
    f.write('average max reward: '+str(ave_max)+', episode: '+str(episode_of_max_ave)+'\n')
    f.write('exection time: '+str(datetime.datetime.now()-start_time))
plt.figure(2)
plt.savefig(save_folder+'/figure.png')
plt.figure(3)
plt.savefig(save_folder+'/loss.png')

print(' start  time : '+str(start_time))
print(' finish time : '+str(datetime.datetime.now()))
print_time(start_time)

print('Complete')
time.sleep(1)
env.render()
env.close()
plt.ioff()
#plt.show()
