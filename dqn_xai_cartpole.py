'''dqn settings'''
EPISODE_NUMBER = 20

'''saliency settings'''
SALIENCY_ROUGHNESS = 4
SALIENCY_MAX = 0.4
#saliency scoreの最大値を予想して設定する、この値をsaliency scoreが超えるとバグる

'''gif dettings'''
SALIENCY_MAP_RATE = 0.8
CONTRAST_MAGNIFICATION = 2.5
SALIENCY_MAX_COLOR = [0,0,255]
SALIENCY_MIN_COLOR = [255,255,255]
FRAME_LENGTH = 160

# -*- coding: utf-8 -*-

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image, ImageDraw, ImageEnhance

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from scipy.ndimage.filters import gaussian_filter
import os
import cv2
import copy


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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

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

    screen_sequence.append(screen/255)

    screen = screen[:, 160:320]
    #縦の、159以下と321以上を捨てる
    #screenの各次元の要素数:(3,160,600)

    view_width = 320
    #NNに入力する画像の幅

    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        #カート<160

        slice_range = slice(view_width)
        #多分160より左をslice_rangeとして指定している

    elif cart_location > (screen_width - view_width // 2):
        #600-160=440<カート

        slice_range = slice(-view_width, None)
        #多分600-160=440より右をslice_rangeとして指定している

    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
        #多分カートの左右160をslice_rangeとして指定している

    # Strip off(はがす) the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    #多分カートの左右160をスクリーンとして切り出している
    #画面外に範囲がはみ出そうなら上記のようにうまくやっている
    #screenの各次元の要素数:(3,160,320)

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    #floatにして、0~255のレンジを0~1にした
    #screenの各次元の要素数:(3,160,320)
    #screenの全要素数153600

    screen = torch.from_numpy(screen)
    #torch.from_numpy(screen)はscreenという名のndarrayからtensorを作成
    #screenという名のテンソルの要素数は153600

    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)
    #次元数1up([3, 160, 320]→[1, 3, 40, 80])
    #screenの全要素数が153600→9600に変化

env.reset()
'''
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()
'''

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = DQN().to(device)
#行動を決めるNNのオブジェクト
#to(device)はgpuとcpuのどちらを使うのか

target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
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
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
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

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

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

    if os.path.exists('images')==False:
        os.mkdir('images')
    #imagesフォルダの作成

    if image.ndim==3:
        image = Image.fromarray(np.uint8(image.transpose(1, 2, 0)))
        #保存は[縦,横,色],0~255
        image.save('images/'+save_name+'.'+save_type)
    elif image.ndim==2:
        image_width = image.shape[1]
        image_hight = image.shape[0]
        output_image = np.zeros((image_hight, image_width, 3), dtype=float)
        for i in range(3):
            output_image[:,:, i] = image
        image = Image.fromarray(np.uint8(output_image))
        #保存は[縦,横,色],0~255
        image.save('images/'+save_name+'.'+save_type)

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
    image3 = copy.copy(image)#np.zeros((3, image_hight, image_width))

    for i in range(3):
        blurred_frame[i] = gaussian_filter(image[i], sigma=blurred_sigma)
        image1[i] = image[i]*(1-mask)
        image2[i] = blurred_frame[i]*mask
        image[i] = image1[i] + image2[i]

    if save==True:
        save_image(image3, 'input_image')
        save_image(mask, 'mask_image'+str(perturbed_point)+',σ='+str(mask_sigma))
        save_image(1-mask, '1-mask_image'+str(perturbed_point)+',σ='+str(mask_sigma))
        save_image(image1, 'input(1-mask_image)'+str(perturbed_point)+',σ='+str(mask_sigma))
        save_image(image2, 'blurred_image_mask_image,σa='+str(blurred_sigma))
        save_image(image, 'perturbed_image'+str(perturbed_point)+',σ='+str(mask_sigma)+',σa='+str(blurred_sigma))
        save_image(blurred_frame, 'blurred_frame,σa='+str(blurred_sigma))

    return image

def make_saliency_map(image, mask_sigma, blurred_sigma, decimation_rate, max_color=[255,255,255], min_color=[0,0,0]):
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
    saliency_map = np.zeros((3, int(state_hight/decimation_rate), int(state_width/decimation_rate) ))
    for i in range(0, state_width, decimation_rate):
        for j in range(0, state_hight, decimation_rate):
            perturbed_state = make_perturbed_image( (image.numpy().squeeze())*255, [j,i], mask_sigma, blurred_sigma)
            perturbed_state = torch.from_numpy(perturbed_state)
            perturbed_state = perturbed_state.unsqueeze(0).to(device)
            perturbed_q = policy_net(perturbed_state)



            #saliency_map[:, int(j/decimation_rate),int(i/decimation_rate)] = ( np.array(max_color)-np.array(min_color) )*( float((normal_q-perturbed_q).pow(2).sum().mul_(0.5)) /255 ) + np.array(min_color)*(1/255)*0.6
            saliency_map[:, int(j/decimation_rate),int(i/decimation_rate)] = (0.1/255)*( (np.array(max_color)-np.array(min_color)) * float((normal_q-perturbed_q).pow(2).sum().mul_(0.5))/SALIENCY_MAX + np.array(min_color) )
            #3色でまとめてsaliencyを計算し、RGBに変換

            if saliency_map[:, int(j/decimation_rate),int(i/decimation_rate)].min()<0:
                print('error:min')

            if saliency_map[:, int(j/decimation_rate),int(i/decimation_rate)].max()>0.1:
                print('error:max')

            if i==0 and j==0:
                print(normal_q)
                print(perturbed_q)
                print((normal_q-perturbed_q).pow(2).sum().mul_(0.5))

            if float((normal_q-perturbed_q).pow(2).sum().mul_(0.5)) > SALIENCY_MAX:
                print('Warning: saliency_score overflowed')
                print('SALIENCY_MAXの値を大きくしてください')


    return saliency_map

num_episodes = EPISODE_NUMBER
saliency_calcuration_rate = SALIENCY_ROUGHNESS

frame_num = 0
#プログラム開始から何フレーム目か

saliency_map_sequence = []
input_sequence = []
screen_sequence = []

for i_episode in range(num_episodes):
    #1エピソード開始

    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    #1エピソード目に行う処理
    if i_episode == 0:
        make_perturbed_image(last_screen.numpy().squeeze(),(20,40),4,3,True)
        #フィルターのサンプル画像を保存

    #1ステップ目開始
    for t in count():

        saliency_map_sequence.append(make_saliency_map(state, 4, 3, saliency_calcuration_rate, SALIENCY_MAX_COLOR, SALIENCY_MIN_COLOR )*10 )
        print( str(saliency_map_sequence[frame_num].min()) +'  '+ str(saliency_map_sequence[frame_num][1].max()) +'  '+str(saliency_map_sequence[frame_num][1,0,0]) )
        #print(saliency_map_sequence[frame_num][1].max())
        input_sequence.append(current_screen.numpy().squeeze())


        frame_num += 1

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

        #1エピソード終了時のdurationのプロット
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())



save_movie(input_sequence, 'input_image')
save_movie(saliency_map_sequence, 'saliency_map')
save_movie(screen_sequence, 'screen', (600,400))
blend_save_movie(saliency_map_sequence, input_sequence, 'synthesis', SALIENCY_MAP_RATE, CONTRAST_MAGNIFICATION,loop=1)

print('Complete')
env.render()
env.close()
plt.ioff()
#plt.show()
