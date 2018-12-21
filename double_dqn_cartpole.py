'''dqn settings'''
EPISODE_NUMBER = 5000

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LEARNING_RATE=0.001

UPDATE_TARGET_Q_FREQ = 3
TRAIN_FREQ = 5

'''saliency settings'''
SALIENCY_SAVING = True #saliency計算するかどうか
SALIENCY_ROUGHNESS = 8

'''ndarray save dettings'''
SAVE_SCREEN = True
SAVE_FREQUENCY = 100 #何エピソードごとに各種画像のndarrayを作成するか
START_DURATION = 20
START_SAVE_FREQUENCY = 10
END_DURATION = 20
END_SAVE_FREQUENCY = 10


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device:gpuとcpuのどちらを使うのか
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
        self.conv1 = nn.Conv2d(3, 64*4, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64*4)
        self.conv2 = nn.Conv2d(64*4, 128*4, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(128*4)
        self.conv3 = nn.Conv2d(128*4, 128*4, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(128*4)
        self.head = nn.Linear(1792*4, 2)

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
    #env.state[0]:カートの座標

def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    #screenという名のndarrayは3次元配列
    #screenの各次元の要素数:(3,400,600)
    #恐らく(色,縦,横),range:0~255

    '''
    if SAVE_SCREEN==True and save_screen==True and i_episode % SAVE_FREQUENCY==0:
        screen_sequence.append(screen/255)
    '''

    screen = screen[:, 160:320]
    #縦の、159以下と321以上を捨てる
    #screenの各次元の要素数:(3,160,600)

    view_width = 320
    #NNに入力する画像の幅

    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off(はがす) the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    #多分カートの左右160をスクリーンとして切り出している
    #画面外に範囲がはみ出そうなら上記のようにうまくやっている
    #screenの各次元の要素数:(3,160,320)

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)
    #次元数1up([3, 160, 320]→[1, 3, 40, 80])
    #screenの全要素数が153600→9600に変化

env.reset()

# BATCH_SIZE = 128
# GAMMA = 0.999
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# TARGET_UPDATE = 10

policy_net = DQN().to(device)
q_ast = deepcopy(policy_net)
#行動を決めるNNのオブジェクト
#to(device)はgpuとcpuのどちらを使うのか
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
    #eps_threshold:ランダムに行動する確率
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():

            #policy_net(state)は各行動のQ値を返す
            return policy_net(state).max(1)[1].view(1, 1)
            #恐らく現在の状態をNNに入力し、出てきた出力をreturnしている
            #恐らくreturnは行動(各行動の確率のベクトルではない)
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
    plt.ylabel('Loss')
    plt.plot(loss_t.numpy())


    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    global last_sync
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
    # # Optimize the model
    # optimizer.zero_grad()
    # loss.backward()
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    # optimizer.step()

def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def save_image(image, save_name, save_type='png'):
    #imageをファイルに保存する
    #image:[色,縦,横]or[縦,横],range:0~1or0~255
    if image.max() <= 1:
        image = image*255
    #rangeの調整
    if os.path.exists(save_folder+'/images')==False:
        os.mkdir(save_folder+'/images')
    #imagesフォルダの作成
    if image.ndim==3:
        image = Image.fromarray(np.uint8(image.transpose(1, 2, 0)))
        #保存は[縦,横,色],0~255
        image.save(save_folder+'/images/'+save_name+'.'+save_type)
    elif image.ndim==2:
        image_width = image.shape[1]
        image_hight = image.shape[0]
        output_image = np.zeros((image_hight, image_width, 3))
        for i in range(3):
            output_image[:,:, i] = image
        image = Image.fromarray(np.uint8(output_image))
        #保存は[縦,横,色],0~255
        image.save(save_folder+'/images/'+save_name+'.'+save_type)


'''ここから'''
def save_movie(image_sequence, save_name, save_size=(640,320), save_type='gif', frame_length=160, loop=0):
    #image_sequence(ndarrayのリスト)から動画を作成し、ファイルに保存する
    #image_sequence:[色,縦,横],range:0~1
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ

    if os.path.exists('images')==False:
        os.mkdir('images')
    #imagesフォルダの作成

    movie = []
    for i in range( len(image_sequence) ):
        movie.append( Image.fromarray(np.uint8(image_sequence[i]*255).transpose(1, 2, 0)))
        movie[i] = movie[i].resize(save_size)

    #保存は[縦,横,色],0~255
    movie[0].save('images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
def blend_save_movie(image_sequence1, image_sequence2, save_name, image_sequence1_rate=0.5, contrast_rate=1, save_type='gif', frame_length=160, loop=0):
    #image_sequence1,2(ndarrayのリスト)を合成した画像から動画を作成し、ファイルに保存する
    #image_sequence:[色,縦,横],range:0~1
    #image_sequence1_rate:大きくすると合成時、image_sequence1の影響が大きくなる,range:0~1
    #contrast_rate:出力gifのコントラストをn倍する
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ

    if len(image_sequence1)!=len(image_sequence2):
        print('error:blend_save_movie')
        print('length of image_sequence1 and image_sequence2 must be same')

    if os.path.exists('images')==False:
        os.mkdir('images')
    #imagesフォルダの作成

    movie = []
    for i in range( len(image_sequence1) ):
        image1 = Image.fromarray(np.uint8(image_sequence1[i]*255).transpose(1, 2, 0))
        image2 = Image.fromarray(np.uint8(image_sequence2[i]*255).transpose(1, 2, 0))
        image1 = image1.resize((640, 320))
        image2 = image2.resize((640, 320))
        image2 = Image.blend(image2, image1, image_sequence1_rate)
        image2 = ImageEnhance.Contrast(image2).enhance( contrast_rate/max(image_sequence1_rate,1-image_sequence1_rate) ) #2)
        movie.append( image2 )

    #保存は[縦,横,色],0~255
    movie[0].save('images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
'''ここまで未使用'''


def save_ndarray_list(ndarray_list, save_name, save_type='npz'):
    print(' outputting '+save_name+'.'+save_type+'...')
    if os.path.exists(save_folder+'/files')==False:
        os.mkdir(save_folder+'/files')
    #filesフォルダの作成
    np.savez_compressed(save_folder+'/files/'+save_name+'.'+save_type, ndarray_list)
    print(' finished')

def make_perturbed_image(image, perturbed_point, mask_sigma, blurred_sigma, save=False):
    #出力:perturbed_image(perturbed_point付近がぼやけた画像)(data:(色,縦,横), dtype:ndarray, shape:(3, hight of image, width of image), range:0~255　))
    #image:(data:(色,縦,横), dtype:ndarray, shape:(3, :, :), range:0~255)
    #perturbed_point:(data:(y座標, x座標))
    #mask_sigma:マスク画像のσ
    #blurred_sigma:ぼやかした画像のσ
    #saveがTrueの時は各画像をファイルに保存

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
        save_image(mask, 'mask_image'+str(perturbed_point)+',σ='+str(mask_sigma))
        save_image(1-mask, '1-mask_image'+str(perturbed_point)+',σ='+str(mask_sigma))
        save_image(image1, 'input(1-mask_image)'+str(perturbed_point)+',σ='+str(mask_sigma))
        save_image(image2, 'blurred_image_mask_image,σa='+str(blurred_sigma))
        save_image(image3, 'perturbed_image'+str(perturbed_point)+',σ='+str(mask_sigma)+',σa='+str(blurred_sigma))
        save_image(blurred_frame, 'blurred_frame,σa='+str(blurred_sigma))

    return image3

def make_saliency_map(image, mask_sigma, blurred_sigma, decimation_rate):
    #saliency_mapを作成(data:(色,縦,横), dtype:ndarray, shape:(3, hight of image, width of image), range:0~0.1　)
    #image:入力画像(data(?, 色, 縦, 横), dtype:tensor, shape:(1, 3, :, :), range:0~1)
    #mask_sigma:マスク画像のσ
    #blurred_sigma:ぼやかした画像のσ
    #decimation_rate:何ピクセルに一回saliencyを計算するか、大きくするとsaliency_mapが粗くなるが、計算が速くなる
    #max_color:saliency_mapの色、saliency_scoreが最大になるときの色を指定する,[R,G,B],range:0~255
    #min_color:saliency_mapの色、saliency_scoreが最小になるときの色を指定する,[R,G,B],range:0^255

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

def decision_of_save(episode_num, average_of_reward, episode_per_saliency_start, episode_per_saliency_duration_start, episode_per_saliency_end, episode_per_saliency_duration_end):
    if EPISODE_NUMBER < episode_per_saliency_duration_start + episode_per_saliency_duration_end:
        print('error:decision_of_save')
        return -1
    saliency_save_flag = False
    if SALIENCY_SAVING == False:
        return saliency_save_flag
    if episode_num < episode_per_saliency_duration_start:
        if episode_num % episode_per_saliency_start == 0:
            saliency_save_flag = True
    elif EPISODE_NUMBER - episode_per_saliency_duration_end <= episode_num:
        if episode_num % episode_per_saliency_end == 0:
            saliency_save_flag = True
    elif episode_num == episode_per_saliency_duration_start or max(average_of_reward[episode_per_saliency_duration_start:-1]) < average_of_reward[-1] or episode_num % SAVE_FREQUENCY == 0:
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
    #NNに入力する画像の幅

    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        position = view_width // 2
    elif cart_location > (screen_width - view_width // 2):
        position = screen_width - view_width // 2
    else:
        position = cart_location
    return position



num_episodes = EPISODE_NUMBER
saliency_calcuration_rate = SALIENCY_ROUGHNESS


saliency_map_sequence = []
input_sequence = []
screen_sequence = []
cart_location_sequence = []
saved_episode = []
saved_episode_rewards = []
ave=0
ave_max=0
average_of_reward=[]
saliency_save_flag = False


start_time = datetime.datetime.now()
print(' start time : '+str(start_time))

save_folder = make_lowest_folder()

#f.close()
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

if SALIENCY_SAVING == True:
    print(' saving variables...')
    variables={"EPISODE_NUMBER":EPISODE_NUMBER, "SAVE_FREQUENCY":SAVE_FREQUENCY, "SALIENCY_ROUGHNESS":SALIENCY_ROUGHNESS, "SAVE_SCREEN":SAVE_SCREEN}
    with open(save_folder+'/files/variables.pickle', mode='wb') as f:
        pickle.dump(variables, f)
    print(' finished')

for i_episode in range(num_episodes):
    #1エピソード開始

    #e_loss
    episode_loss = []
    # Initialize the environment and state
    env.reset()
    #save_screen = False #初期値取得の時は保存しない
    last_screen = get_screen()
    current_screen = get_screen()
    #save_screen = True
    state = current_screen - last_screen

    if ave > ave_max and i_episode >= START_DURATION+1:
        ave_max = ave
        episode_of_max_ave = i_episode

    ave_q.append(np.mean(policy_net(state).cpu().detach().numpy()))

    #1ステップ目開始
    for t in count():

        #1エピソード目に行う処理
        if i_episode == 0 and t == 3:
            make_perturbed_image(state.squeeze().cpu().numpy(),(20,40),4,3,save=True)
            print(' filter images were saved')
            #フィルターのサンプル画像を保存

        if saliency_save_flag == True: #i_episode=0,5,10...=1,6,11...エピソード目
            saliency_map_sequence.append( make_saliency_map(state, 4, 3, saliency_calcuration_rate ) )
            input_sequence.append( current_screen.cpu().numpy().squeeze() )
            if SAVE_SCREEN==True:
                screen_sequence.append(env.render(mode='rgb_array').transpose((2, 0, 1))/255)
                cart_location_sequence.append(get_input_position())

        # Select and perform an action
        action = select_action(state)
        #action:0が左で1が右に動かす
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

        #1エピソード終了時の動作
        if done:

            episode_durations.append(t + 1)
            #episode_loss.append(e_loss)

            ave = mean(episode_durations)
            average_of_reward.append(mean(episode_durations))

            if saliency_save_flag==True:

                saliency_map_sequence.append(np.full_like(saliency_map_sequence[0], -1))
                input_sequence.append(np.full_like(input_sequence[0], -1))
                if SAVE_SCREEN==True:
                    screen_sequence.append(np.full_like(screen_sequence[0], -1))
                    cart_location_sequence.append(-1)
                #1エピソードの終わりに、-1で埋め尽くしたndarrayをいれる

                saved_episode.append(i_episode)
                saved_episode_rewards.append(average_of_reward[-1])
                print(' episode: '+str(i_episode)+' / '+str(EPISODE_NUMBER-1)+', reward: '+str(t+1)+', average/ave_max: '+f'{ave:.2f}'+'/'+f'{ave_max:.2f}'+' saliency was generated')
            else:
                print(' episode: '+str(i_episode)+' / '+str(EPISODE_NUMBER-1)+', reward: '+str(t+1)+', average/ave_max: '+f'{ave:.2f}'+'/'+f'{ave_max:.2f}')
            saliency_save_flag = decision_of_save(i_episode, average_of_reward, START_SAVE_FREQUENCY, START_DURATION, END_SAVE_FREQUENCY, END_DURATION)
            ave_loss.append(mean(episode_loss)/(i_episode+1))

            if (i_episode+1) % 10 == 0:
                plt.figure(2)
                plt.savefig(save_folder+'/figure.png')
                plt.figure(3)
                plt.savefig(save_folder+'/loss.png')
                if SALIENCY_SAVING == True:
                    save_ndarray_list(input_sequence, 'input')
                    save_ndarray_list(saliency_map_sequence, 'saliency_map')
                    save_ndarray_list(saved_episode, 'episodes')
                    save_ndarray_list(saved_episode_rewards, 'rewards')
                    if SAVE_SCREEN==True:
                        save_ndarray_list(screen_sequence, 'screen')
                        save_ndarray_list(cart_location_sequence, 'cart_location')

                print_time(start_time)
                plot_durations()
                plot_loss()
            break

    # Update the target network
    #if i_episode % TARGET_UPDATE == 0:
    #    target_net.load_state_dict(policy_net.state_dict())


#print(cart_location_sequence)

print(' number of saved episode : '+str(len(saved_episode)))
print(' saved episode number : '+str(saved_episode))
print(saved_episode_rewards)


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
plt.savefig(save_folder+'/figure.png')

if SALIENCY_SAVING == True:
    save_ndarray_list(input_sequence, 'input')
    save_ndarray_list(saliency_map_sequence, 'saliency_map')
    save_ndarray_list(saved_episode, 'episodes')
    save_ndarray_list(saved_episode_rewards, 'rewards')
    if SAVE_SCREEN==True:
        save_ndarray_list(screen_sequence, 'screen')
        save_ndarray_list(cart_location_sequence, 'cart_location')

    print(' saving variables...')
    variables={"EPISODE_NUMBER":EPISODE_NUMBER, "SAVE_FREQUENCY":SAVE_FREQUENCY, "SALIENCY_ROUGHNESS":SALIENCY_ROUGHNESS, "SAVE_SCREEN":SAVE_SCREEN}
    with open(save_folder+'/files/variables.pickle', mode='wb') as f:
        pickle.dump(variables, f)
    print(' finished')


print(' start  time : '+str(start_time))
print(' finish time : '+str(datetime.datetime.now()))
print_time(start_time)

print('Complete')
time.sleep(1)
env.render()
env.close()
plt.ioff()
#plt.show()
