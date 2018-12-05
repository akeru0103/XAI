'''gif dettings'''
FRAME_PER_GIF = 100 #1つのgifファイルに入れるフレーム数の最大値。大きくすると負荷がかかる
SAVE_SCREEN = False
SALIENCY_MAP_RATE = 0.7
CONTRAST_MAGNIFICATION = 2.5
SALIENCY_MAX_COLOR = [0,0,255]
SALIENCY_MIN_COLOR = [255,255,255]
FRAME_LENGTH = 160



import numpy as np
import os
from PIL import Image, ImageDraw, ImageEnhance
import pickle
from itertools import count

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

def save_movie_from_list(image_sequence, save_name, save_size=(640,320), save_type='gif', frame_length=160, loop=0):
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

def load_ndarray(file_name, file_type='npz'):
    loaded_array = np.load('files/'+file_name+'.'+file_type)
    return loaded_array['arr_0']

def range_change(input, i_range, o_range):
    #値の範囲を変える
    #i_range:(inputの最小値, inputの最大値)
    #o_range:(outputの最大値, outputの最大値)
    output = ( (o_range[1]-o_range[0])*input + (i_range[1]*o_range[0]-o_range[1]*i_range[0]) )/(i_range[1]-i_range[0])
    return output

def mono_to_color(ndarray, ndarray_max, ndarray_min=0, max_color=[0,0,0], min_color=[255,255,255]):
    #ndarrayを全ndarray_maxとndarray_minに正規化した後にrgb情報を追加(data:(色,縦,横), dtype:ndarray, shape:(3, hight of ndarray, width of ndarray), range:0~1　)
    #ndarray:入力2次元配列(data(縦, 横), dtype:ndarray, shape:(:, :), range:なし)
    #max_color:saliency_mapの色、saliency_scoreが最大になるときの色を指定する,[R,G,B],range:0~255
    #min_color:saliency_mapの色、saliency_scoreが最小になるときの色を指定する,[R,G,B],range:0^255

    ndarray_width = ndarray.shape[1]
    ndarray_hight = ndarray.shape[0]
    color_ndarray = np.zeros((3, ndarray_hight, ndarray_width ))
    for i in range(0, ndarray_width):
        for j in range(0, ndarray_hight):

            ndarray[j,i] = range_change(ndarray[j,i], (ndarray_min,ndarray_max), (0,1))
            #saliency_map[:, int(j/decimation_rate),int(i/decimation_rate)] = ( np.array(max_color)-np.array(min_color) )*( float((normal_q-perturbed_q).pow(2).sum().mul_(0.5)) /255 ) + np.array(min_color)*(1/255)*0.6
            color_ndarray[:, j, i] = (1/255)*( (np.array(max_color)-np.array(min_color)) * ndarray[j, i] + np.array(min_color) )
            #3色でまとめてsaliencyを計算し、RGBに変換

    return color_ndarray

def save_movie_from_ndarray(image_sequence, save_name, save_size=(640,320), save_type='gif', frame_length=160, loop=0, max_color=[0,0,0], min_color=[255,255,255]):
    #image_sequence(ndarray)から動画を作成し、ファイルに保存する
    #image_sequence:[フレーム数,縦,横]or[フレーム数,色,縦,横],range:0~1
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ

    if os.path.exists('images')==False:
        os.mkdir('images')
    #imagesフォルダの作成

    movie = []
    array_max = image_sequence.max()
    array_min = image_sequence.min()
    for i in range( image_sequence.shape[0] ):
        if image_sequence.ndim==3:
            movie.append( Image.fromarray(np.uint8( mono_to_color(image_sequence[i], array_max, array_min, max_color, min_color) *255).transpose(1, 2, 0)))
        else:
            movie.append( Image.fromarray(np.uint8(image_sequence[i]*255).transpose(1, 2, 0)))
        movie[i] = movie[i].resize(save_size)

    #保存は[縦,横,色],0~255
    movie[0].save('images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)

def blend_save_movie_from_ndarray(image_sequence1, image_sequence2, save_name, save_size=(640,320), image_sequence1_rate=0.5, contrast_rate=1, save_type='gif', frame_length=160, loop=0, max_color1=[0,0,0], min_color1=[255,255,255], max_color2=[0,0,0], min_color2=[255,255,255]):
    #image_sequence1,2(ndarray)を合成した画像から動画を作成し、ファイルに保存する
    #image_sequence:[フレーム数,色,縦,横],range:0~1
    #image_sequence1_rate:大きくすると合成時、image_sequence1の影響が大きくなる,range:0~1
    #contrast_rate:出力gifのコントラストをn倍する
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ

    if image_sequence1.shape[0]!=image_sequence2.shape[0]:
        print('error:blend_save_movie')
        print('length of image_sequence1 and image_sequence2 must be same')
        return -1

    if os.path.exists('images')==False:
        os.mkdir('images')
    #imagesフォルダの作成

    movie = []
    array1_max = image_sequence1.max()
    array1_min = image_sequence1.min()
    array2_max = image_sequence2.max()
    array2_min = image_sequence2.min()
    for i in range( len(image_sequence1) ):
        if image_sequence1.ndim==3:
            image1 = Image.fromarray(np.uint8( mono_to_color(image_sequence1[i], array1_max, array1_min, max_color1, min_color1) *255).transpose(1, 2, 0))
        else:
            image1 = Image.fromarray(np.uint8(image_sequence1[i]*255).transpose(1, 2, 0))

        if image_sequence2.ndim==3:
            image2 = Image.fromarray(np.uint8( mono_to_color(image_sequence2[i], array2_max, array2_min, max_color2, min_color2) *255).transpose(1, 2, 0))
        else:
            image2 = Image.fromarray(np.uint8(image_sequence2[i]*255).transpose(1, 2, 0))

        #image1 = Image.fromarray(np.uint8(image_sequence1[i]*255).transpose(1, 2, 0))
        #image2 = Image.fromarray(np.uint8(image_sequence2[i]*255).transpose(1, 2, 0))
        image1 = image1.resize(save_size)
        image2 = image2.resize(save_size)
        image2 = Image.blend(image2, image1, image_sequence1_rate)
        image2 = ImageEnhance.Contrast(image2).enhance( contrast_rate/max(image_sequence1_rate,1-image_sequence1_rate) ) #2)
        movie.append( image2 )

    #保存は[縦,横,色],0~255
    movie[0].save('images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)

def divide_ndarray_and_save_movie(image_sequence, save_name, frame_per_gif, save_size=(640,320), save_type='gif', frame_length=160, loop=0, max_color=[0,0,0], min_color=[255,255,255]):
    #ndarrayをframe_per_gifにいい感じに区切ってgifを作成し、ファイルに保存する
    #image_sequence:[フレーム数,縦,横]or[フレーム数,色,縦,横],range:0~1
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ
    print(' generating '+save_name+'.'+save_type+'...')

    if os.path.exists('images')==False:
        os.mkdir('images')
    #imagesフォルダの作成

    array_max = image_sequence.max()
    array_min = 0#image_sequence.min()

    from_frame = 0
    from_episode = 0
    last_flag = False

    for j in count():
        skip_flag = False
        movie = []
        movie_position = 0
        to_frame = from_frame
        to_episode = from_episode
        #print('A, from:'+str(from_frame)+', to:'+str(to_frame))
        to_frame = to_frame + frame_per_gif -1
        #print('B, from:'+str(from_frame)+', to:'+str(to_frame))

        '''from_frameがndarrayからはみ出た'''
        if from_frame >= image_sequence.shape[0]:
            break

        '''to_frameがndarrayからはみ出た'''
        if to_frame >= image_sequence.shape[0] - 1:
            last_flag = True
            to_frame = image_sequence.shape[0] - 1

        for i in count():
            if from_frame == to_frame:
                print('frame of episode'+str(from_episode)+' > FRAME_PER_GIF')
                print('making gif of episode '+str(from_episode)+' was skipped')
                skip_flag = True
                break
            if image_sequence[to_frame].max()==-1:
                break
            to_frame = to_frame - 1

        if skip_flag==True:
            for i in count():
                if image_sequence[from_frame].max()==-1:
                    break
                from_frame = from_frame + 1
            from_frame = from_frame + 1
            from_episode = from_episode + 1
            continue
        #print('C, from:'+str(from_frame)+', to:'+str(to_frame))
        for i in range(from_frame, to_frame + 1):#0~99
            #print(i)
            #print(image_sequence[i,0,0,0])
            if image_sequence.ndim==3:
                if image_sequence[i,0,0]==-1:
                    to_episode = to_episode + 1
                    movie.append( Image.fromarray(np.uint8( np.full((3, image_sequence.shape[1], image_sequence.shape[2]),255).transpose(1, 2, 0))))
                else:
                    movie.append( Image.fromarray(np.uint8( mono_to_color(image_sequence[i], array_max, array_min, max_color, min_color) *255).transpose(1, 2, 0)))
            else:
                if image_sequence[i,0,0,0]==-1:
                    to_episode = to_episode + 1
                    #print('a')
                    #print(i)
                    movie.append( Image.fromarray(np.uint8( np.full_like(image_sequence[0],255).transpose(1, 2, 0))))
                else:
                    movie.append( Image.fromarray(np.uint8(image_sequence[i]*255).transpose(1, 2, 0)))
            movie[movie_position] = movie[movie_position].resize(save_size)
            movie_position = movie_position + 1
        #保存は[縦,横,色],0~255
        movie[0].save('images/'+save_name+',episode'+str(from_episode*variables["SAVE_FREQUENCY"])+'-'+str((to_episode-1)*variables["SAVE_FREQUENCY"])+',every'+str(variables["SAVE_FREQUENCY"])+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
        from_frame = to_frame + 1
        from_episode = to_episode
        #print('D, from:'+str(from_frame)+', to:'+str(to_frame))
        if last_flag==True:
            break
    print(' finished')

def divide_and_blend_ndarray_and_save_movie(image_sequence1, image_sequence2, save_name, frame_per_gif, image_sequence1_rate=0.5, contrast_rate=1, save_size=(640,320), save_type='gif', frame_length=160, loop=0, max_color=[0,0,0], min_color=[255,255,255]):
    #ndarrayをframe_per_gifにいい感じに区切ってgifを作成し、ファイルに保存する
    #image_sequence:[フレーム数,縦,横]or[フレーム数,色,縦,横],range:0~1
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ
    print(' generating '+save_name+'.'+save_type+'...')

    if image_sequence1.shape[0]!=image_sequence2.shape[0]:
        print('error:blend_save_movie')
        print('length of image_sequence1 and image_sequence2 must be same')
        return -1

    if os.path.exists('images')==False:
        os.mkdir('images')
    #imagesフォルダの作成

    array1_max = image_sequence1.max()
    array1_min = 0#image_sequence1.min()
    array2_max = image_sequence2.max()
    array2_min = 0#image_sequence2.min()

    from_frame = 0
    from_episode = 0
    last_flag = False

    for j in count():
        skip_flag = False
        movie = []
        movie_position = 0
        to_frame = from_frame
        to_episode = from_episode
        #print('A, from:'+str(from_frame)+', to:'+str(to_frame))
        to_frame = to_frame + frame_per_gif -1
        #print('B, from:'+str(from_frame)+', to:'+str(to_frame))

        '''from_frameがndarrayからはみ出た'''
        if from_frame >= image_sequence1.shape[0]:
            break

        '''to_frameがndarrayからはみ出た'''
        if to_frame >= image_sequence1.shape[0] - 1:
            last_flag = True
            to_frame = image_sequence1.shape[0] - 1

        for i in count():
            if from_frame == to_frame:
                print('frame of episode'+str(from_episode)+' > FRAME_PER_GIF')
                print('making gif of episode '+str(from_episode)+' was skipped')
                skip_flag = True
                break
            if image_sequence1[to_frame].max()==-1:
                break
            to_frame = to_frame - 1

        if skip_flag==True:
            for i in count():
                if image_sequence1[from_frame].max()==-1:
                    break
                from_frame = from_frame + 1
            from_frame = from_frame + 1
            from_episode = from_episode + 1
            continue
        #print('C, from:'+str(from_frame)+', to:'+str(to_frame))
        for i in range(from_frame, to_frame + 1):#0~99
            #print(i)
            #print(image_sequence[i,0,0,0])
            if image_sequence1.ndim==3:
                if image_sequence1[i,0,0]==-1:
                    to_episode = to_episode + 1
                    image1 = Image.fromarray(np.uint8( np.full((3, image_sequence1.shape[1], image_sequence1.shape[2]),255).transpose(1, 2, 0)))
                else:
                    image1 = Image.fromarray(np.uint8( mono_to_color(image_sequence1[i], array1_max, array1_min, max_color, min_color) *255).transpose(1, 2, 0))
            else:
                if image_sequence1[i,0,0,0]==-1:
                    to_episode = to_episode + 1
                    #print('a')
                    #print(i)
                    image1 = Image.fromarray(np.uint8( np.full_like(image_sequence1[0],255).transpose(1, 2, 0)))
                else:
                    image1 = Image.fromarray(np.uint8(image_sequence1[i]*255).transpose(1, 2, 0))

            if image_sequence2.ndim==3:
                if image_sequence2[i,0,0]==-1:
                    #to_episode = to_episode + 1
                    image2 = Image.fromarray(np.uint8( np.full((3, image_sequence2.shape[1], image_sequence2.shape[2]),255).transpose(1, 2, 0)))
                else:
                    image2 = Image.fromarray(np.uint8( mono_to_color(image_sequence2[i], array2_max, array2_min, max_color, min_color) *255).transpose(1, 2, 0))
            else:
                if image_sequence2[i,0,0,0]==-1:
                    #to_episode = to_episode + 1
                    #print('a')
                    #print(i)
                    image2 = Image.fromarray(np.uint8( np.full_like(image_sequence2[0],255).transpose(1, 2, 0)))
                else:
                    image2 = Image.fromarray(np.uint8(image_sequence2[i]*255).transpose(1, 2, 0))

            #movie[movie_position] = movie[movie_position].resize(save_size)
            image1 = image1.resize((640, 320))
            image2 = image2.resize((640, 320))
            image2 = Image.blend(image1, image2, image_sequence1_rate)
            image2 = ImageEnhance.Contrast(image2).enhance( contrast_rate/max(image_sequence1_rate,1-image_sequence1_rate) ) #2)
            movie.append( image2 )
            movie_position = movie_position + 1
        #保存は[縦,横,色],0~255
        movie[0].save('images/'+save_name+',episode'+str(from_episode*variables["SAVE_FREQUENCY"])+'-'+str((to_episode-1)*variables["SAVE_FREQUENCY"])+',every'+str(variables["SAVE_FREQUENCY"])+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
        from_frame = to_frame + 1
        from_episode = to_episode
        #print('D, from:'+str(from_frame)+', to:'+str(to_frame))
        if last_flag==True:
            break
    print(' finished')

def divide_ndarray_every_episode(image_sequence):
    print(' dividing every episode...')
    from_frame = 0
    movies = []
    for i in count():
        '''from_frameがndarrayからはみ出た'''
        if from_frame >= image_sequence.shape[0]:
            break
        a_movie = []
        for j in count():
            if image_sequence[from_frame+j].max()==-1:
                from_frame = from_frame + j + 1
                break
            a_movie.append(image_sequence[from_frame+j])
        movies.append(np.array(a_movie))
    print(' finished')
    return movies

def save_movies_from_ndarray_list(image_sequence_list, episode_number_ndarray, reward_ndarray, save_name, save_size=(640,320), save_type='gif', frame_length=160, loop=0, max_color=[0,0,0], min_color=[255,255,255]):
    print(' generating '+save_name+'.'+save_type+'...')
    digit = len(str(episode_number_ndarray[-1]))
    for i, a_movie in enumerate(image_sequence_list):
        episode_num = str(episode_number_ndarray[i]).rjust(digit,'0')
        save_movie_from_ndarray(a_movie, save_name+' ,epi'+episode_num+' ,ave_rew'+f'{reward_ndarray[i]:.2f}', save_size=save_size, save_type=save_type, frame_length=frame_length, loop=loop, max_color=max_color, min_color=min_color)
    print(' finished')

def blend_save_movies_from_ndarray_lists(image_sequence_list1, image_sequence_list2, episode_number_ndarray, reward_ndarray, save_name, image_sequence1_rate=0.5, contrast_rate=1, save_size=(640,320), save_type='gif', frame_length=160, loop=0, max_color1=[0,0,0], min_color1=[255,255,255],  max_color2=[0,0,0], min_color2=[255,255,255]):
    print(' generating '+save_name+'.'+save_type+'...')
    digit = len(str(episode_number_ndarray[-1]))

    for i, (a_movie1,a_movie2) in enumerate(zip(image_sequence_list1,image_sequence_list2)):
        episode_num = str(episode_number_ndarray[i]).rjust(digit,'0')
        blend_save_movie_from_ndarray(a_movie1, a_movie2, save_name+' ,epi'+episode_num+' ,ave_rew'+f'{reward_ndarray[i]:.2f}', image_sequence1_rate=image_sequence1_rate, contrast_rate=contrast_rate, save_size=save_size, save_type=save_type, frame_length=frame_length, loop=loop, max_color1=max_color1, min_color1=min_color1, max_color2=max_color2, min_color2=min_color2)
    print(' finished')


with open('files/variables.pickle', mode='rb') as f:
    variables=pickle.load(f)

print(variables)


print(' loading ndarrays...')
input_sequence = load_ndarray('input')
saliency_map_sequence = load_ndarray('saliency_map')
saved_episode = load_ndarray('episodes')
saved_episode_rewards = load_ndarray('rewards')
if SAVE_SCREEN==True and variables["SAVE_SCREEN"]==True:
    screen_sequence = load_ndarray('screen')
print(' finished')


input_sequence_list = divide_ndarray_every_episode(input_sequence)
saliency_map_sequence_list = divide_ndarray_every_episode(saliency_map_sequence)
if SAVE_SCREEN==True and variables["SAVE_SCREEN"]==True:
    screen_sequence_list = divide_ndarray_every_episode(screen_sequence)

#print(len(input_sequence_list))

save_movies_from_ndarray_list(input_sequence_list, saved_episode, saved_episode_rewards, 'input', loop=1)
save_movies_from_ndarray_list(saliency_map_sequence_list, saved_episode, saved_episode_rewards, 'saliency', loop=1, max_color=SALIENCY_MAX_COLOR, min_color=SALIENCY_MIN_COLOR)
blend_save_movies_from_ndarray_lists(saliency_map_sequence_list, input_sequence_list, saved_episode, saved_episode_rewards, 'synthesis', loop=0, max_color1=SALIENCY_MAX_COLOR, min_color1=SALIENCY_MIN_COLOR, image_sequence1_rate=SALIENCY_MAP_RATE, contrast_rate=CONTRAST_MAGNIFICATION)

'''
divide_ndarray_and_save_movie(input_sequence, 'input', FRAME_PER_GIF, loop=1)
divide_ndarray_and_save_movie(saliency_map_sequence, 'saliency_map', FRAME_PER_GIF, loop=1, max_color=SALIENCY_MAX_COLOR, min_color=SALIENCY_MIN_COLOR)
if SAVE_SCREEN==True and variables["SAVE_SCREEN"]==True:
    divide_ndarray_and_save_movie(screen_sequence, 'screen', FRAME_PER_GIF, loop=1)

divide_and_blend_ndarray_and_save_movie(input_sequence, saliency_map_sequence, 'synthesis', FRAME_PER_GIF, loop=1, max_color=SALIENCY_MAX_COLOR, min_color=SALIENCY_MIN_COLOR, image_sequence1_rate=SALIENCY_MAP_RATE, contrast_rate=CONTRAST_MAGNIFICATION)
'''

'''
input_sequence[5]=np.full_like(input_sequence[0], -1)
input_sequence[8]=np.full_like(input_sequence[0], -1)
input_sequence[12]=np.full_like(input_sequence[0], -1)
input_sequence[16]=np.full_like(input_sequence[0], -1)
input_sequence[21]=np.full_like(input_sequence[0], -1)
input_sequence[23]=np.full_like(input_sequence[0], -1)


divide_ndarray_and_save_movie(input_sequence, 'test', 10, loop=1)
#
'''


'''
save_movie_from_ndarray(input_sequence, 'input')
save_movie_from_ndarray(saliency_map_sequence, 'saliency_map', max_color=[255,0,0])
save_movie_from_ndarray(screen_sequence, 'screen', save_size=(600,400))
blend_save_movie_from_ndarray(input_sequence, saliency_map_sequence, 'synthesis', loop=1)
'''
