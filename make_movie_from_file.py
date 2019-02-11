'''gif dettings'''
RESULT_NUMBER = 0 #処理するresultフォルダの番号、0だと未処理のフォルダで一番数字の大きいフォルダになる
FRAME_PER_GIF = 100 #1つのgifファイルに入れるフレーム数の最大値。大きくすると負荷がかかる
SAVE_SCREEN = True
SALIENCY_MAP_RATE = 0.7
CONTRAST_MAGNIFICATION = 2.5
SALIENCY_MAX_COLOR = [0,0,255]
SALIENCY_MIN_COLOR = [255,255,255]
FRAME_LENGTH = 120#160


import numpy as np
import os
from PIL import Image, ImageDraw, ImageEnhance
import pickle
from itertools import count
import sys

'''ここから'''
def save_image(image, save_name, save_type='png'):
    #imageをファイルに保存する
    #image:[色,縦,横]or[縦,横],range:0~1or0~255

    if image.max() <= 1:
        image = image*255
    #rangeの調整

    if os.path.exists('results/result'+str(RESULT_NUMBER)+'/images')==False:
        os.mkdir('results/result'+str(RESULT_NUMBER)+'/images')
    #imagesフォルダの作成

    if image.ndim==3:
        image = Image.fromarray(np.uint8(image.transpose(1, 2, 0)))
        #保存は[縦,横,色],0~255
        image.save('results/result'+str(RESULT_NUMBER)+'/images/'+save_name+'.'+save_type)
    elif image.ndim==2:
        image_width = image.shape[1]
        image_hight = image.shape[0]
        output_image = np.zeros((image_hight, image_width, 3), dtype=float)
        for i in range(3):
            output_image[:,:, i] = image
        image = Image.fromarray(np.uint8(output_image))
        #保存は[縦,横,色],0~255
        image.save('results/result'+str(RESULT_NUMBER)+'/images/'+save_name+'.'+save_type)
def save_movie_from_list(image_sequence, save_name, save_size=(640,320), save_type='gif', frame_length=160, loop=0):
    #image_sequence(ndarrayのリスト)から動画を作成し、ファイルに保存する
    #image_sequence:[色,縦,横],range:0~1
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ

    if os.path.exists('results/result'+str(RESULT_NUMBER)+'/images')==False:
        os.mkdir('results/result'+str(RESULT_NUMBER)+'/images')
    #imagesフォルダの作成

    movie = []
    for i in range( len(image_sequence) ):
        movie.append( Image.fromarray(np.uint8(image_sequence[i]*255).transpose(1, 2, 0)))
        movie[i] = movie[i].resize(save_size)

    #保存は[縦,横,色],0~255
    movie[0].save('results/result'+str(RESULT_NUMBER)+'/images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
def divide_ndarray_and_save_movie(image_sequence, save_name, frame_per_gif, save_size=(640,320), save_type='gif', frame_length=160, loop=0, max_color=[0,0,0], min_color=[255,255,255]):
    #ndarrayをframe_per_gifにいい感じに区切ってgifを作成し、ファイルに保存する
    #image_sequence:[フレーム数,縦,横]or[フレーム数,色,縦,横],range:0~1
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ
    print(' generating '+save_name+'.'+save_type+'...')

    if os.path.exists('results/result'+str(RESULT_NUMBER)+'/images')==False:
        os.mkdir('results/result'+str(RESULT_NUMBER)+'/images')
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

        #from_frameがndarrayからはみ出た
        if from_frame >= image_sequence.shape[0]:
            break

        #to_frameがndarrayからはみ出た
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
        movie[0].save('results/result'+str(RESULT_NUMBER)+'/images/'+save_name+',episode'+str(from_episode*variables2["SAVE_FREQUENCY"])+'-'+str((to_episode-1)*variables2["SAVE_FREQUENCY"])+',every'+str(variables2["SAVE_FREQUENCY"])+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
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

    if os.path.exists('results/result'+str(RESULT_NUMBER)+'/images')==False:
        os.mkdir('results/result'+str(RESULT_NUMBER)+'/images')
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

        #from_frameがndarrayからはみ出た
        if from_frame >= image_sequence1.shape[0]:
            break

        #to_frameがndarrayからはみ出た
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
        movie[0].save('results/result'+str(RESULT_NUMBER)+'/images/'+save_name+',episode'+str(from_episode*variables2["SAVE_FREQUENCY"])+'-'+str((to_episode-1)*variables2["SAVE_FREQUENCY"])+',every'+str(variables2["SAVE_FREQUENCY"])+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
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
        #from_frameがndarrayからはみ出た
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
def blend_save_movies_from_ndarray_lists_different_size(image_sequence_list1, image_sequence_list2, position_sequence_list, episode_number_ndarray, reward_ndarray, save_name, image_sequence1_rate=0.5, contrast_rate=1, save_size1=(640,320), save_size2=(640,320), save_type='gif', frame_length=160, loop=0, max_color1=[0,0,0], min_color1=[255,255,255],  max_color2=[0,0,0], min_color2=[255,255,255]):
    print(' generating '+save_name+'.'+save_type+'...')
    digit = len(str(episode_number_ndarray[-1]))

    for i, (a_movie1,a_movie2,position_sequence) in enumerate(zip(image_sequence_list1,image_sequence_list2,position_sequence_list)):
        episode_num = str(episode_number_ndarray[i]).rjust(digit,'0')
        blend_save_movie_from_ndarray_different_size(a_movie1, a_movie2, position_sequence, save_name+' ,epi'+episode_num+' ,ave_rew'+f'{reward_ndarray[i]:.2f}', image_sequence1_rate=image_sequence1_rate, contrast_rate=contrast_rate, save_size1=save_size1, save_size2=save_size2, save_type=save_type, frame_length=frame_length, loop=loop, max_color1=max_color1, min_color1=min_color1, max_color2=max_color2, min_color2=min_color2)
    print(' finished')
'''ここまで未使用'''

def load_ndarray(file_name, file_type='npz'):
    print('   loading '+file_name+'.'+file_type+'...')
    loaded_array = np.load('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/files/'+file_name+'.'+file_type)
    #print(' finished')
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

def save_movie_from_ndarray(image_sequence, save_name, expansion_rate=1, save_size=(640,320), save_type='gif', frame_length=160, loop=0, max_color=[0,0,0], min_color=[255,255,255]):
    #image_sequence(ndarray)から動画を作成し、ファイルに保存する
    #image_sequence:[フレーム数,縦,横]or[フレーム数,色,縦,横],range:0~1
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ

    if os.path.exists('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/images')==False:
        os.mkdir('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/images')
    #imagesフォルダの作成

    movie = []
    array_max = image_sequence.max()
    array_min = image_sequence.min()
    for i in range( image_sequence.shape[0] ):
        if image_sequence.ndim==3:
            movie.append( Image.fromarray(np.uint8( mono_to_color(image_sequence[i], array_max, array_min, max_color, min_color) *255).transpose(1, 2, 0)))
        else:
            movie.append( Image.fromarray(np.uint8(image_sequence[i]*255).transpose(1, 2, 0)))
        movie[i] = movie[i].resize( (int(movie[i].width*expansion_rate), int(movie[i].height*expansion_rate)) )

    #保存は[縦,横,色],0~255
    movie[0].save('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
def blend_save_movie_from_ndarray(image_sequence1, image_sequence2, save_name, expansion_rate=1, save_size=(640,320), image_sequence1_rate=0.5, contrast_rate=1, save_type='gif', frame_length=160, loop=0, max_color1=[0,0,0], min_color1=[255,255,255], max_color2=[0,0,0], min_color2=[255,255,255]):
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

    if os.path.exists('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/images')==False:
        os.mkdir('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/images')
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
        #print(np.array(image1).shape)
        #print(np.array(image2).shape)
        image2 = image2.resize( (int(image2.width*expansion_rate), int(image2.height*expansion_rate)) )
        image1 = image1.resize( (image2.width, image2.height) )
        #print(np.array(image1).shape)
        #print(np.array(image2).shape)
        image2 = Image.blend(image2, image1, image_sequence1_rate)
        image2 = ImageEnhance.Contrast(image2).enhance( contrast_rate/max(image_sequence1_rate,1-image_sequence1_rate) ) #2)
        movie.append( image2 )

    #保存は[縦,横,色],0~255
    movie[0].save('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
def blend_save_movie_from_ndarray_different_size(image_sequence1, image_sequence2, position_sequence, save_name, expansion_rate=1, save_size1=(640,320), save_size2=(640,320), image_sequence1_rate=0.5, contrast_rate=1, save_type='gif', frame_length=160, loop=0, max_color1=[0,0,0], min_color1=[255,255,255], max_color2=[0,0,0], min_color2=[255,255,255]):
    #image_sequence1,2(ndarray)を合成した画像から動画を作成し、ファイルに保存する
    #image_sequence:[フレーム数,色,縦,横],range:0~1
    #image_sequence1_rate:大きくすると合成時、image_sequence1の影響が大きくなる,range:0~1
    #contrast_rate:出力gifのコントラストをn倍する
    #frame_length:1フレームあたりの表示する時間
    #loop:何回ループするか,0だと無限ループ

    if image_sequence1.shape[0]!=image_sequence2.shape[0] or image_sequence1.shape[0]!=len(position_sequence):
        print('error:blend_save_movie')
        print('length of image_sequence1 and image_sequence2 must be same')
        return -1

    if os.path.exists('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/images')==False:
        os.mkdir('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/images')
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

        #image1 = image1.resize(save_size1)
        #image2 = image2.resize(save_size2)
        image2 = image2.resize( (int(image2.width*expansion_rate), int(image2.height*expansion_rate)) )
        image1 = image1.resize( (int(320*expansion_rate), int(160*expansion_rate)) )
        image1 = paste_image_to_background(image1, (image2.width,image2.height), (int((position_sequence[i]-160)*expansion_rate),int(160*expansion_rate)) )
        image2 = Image.blend(image2, image1, image_sequence1_rate)
        image2 = ImageEnhance.Contrast(image2).enhance( contrast_rate/max(image_sequence1_rate,1-image_sequence1_rate) ) #2)
        movie.append( image2 )

    #movie_sadfadfas[0].save('results/result'+str(RESULT_NUMBER)+'/images/sadfadfas.'+save_type, save_all=True, append_images=movie_sadfadfas[1:], optimize=False, duration=frame_length, loop=loop)

    #保存は[縦,横,色],0~255
    movie[0].save('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/images/'+save_name+'.'+save_type, save_all=True, append_images=movie[1:], optimize=False, duration=frame_length, loop=loop)
def paste_image_to_background(image, background_size, paste_position):
    background = np.full((3,background_size[1],background_size[0]), 1)
    background = Image.fromarray(np.uint8(background*255).transpose(1, 2, 0))
    background.paste(image, paste_position)
    return background

def save_movie_from_npz(npz_name, save_name, expansion_rate=1, data_type='npz', frame_length=160):
    loaded_array = load_ndarray(npz_name)
    print('   saving '+save_name+'.gif...')
    save_movie_from_ndarray(loaded_array, save_name, expansion_rate=expansion_rate, frame_length=frame_length)
def blend_save_movie_from_npz(npz_name1, npz_name2, save_name, expansion_rate=1, image1_rate=0.5, contrast_rate=1, save_type='gif', frame_length=160, loop=0, max_color1=[0,0,0], min_color1=[255,255,255], max_color2=[0,0,0], min_color2=[255,255,255]):
    loaded_array1 = load_ndarray(npz_name1)
    loaded_array2 = load_ndarray(npz_name2)
    print('   saving '+save_name+'.gif...')
    blend_save_movie_from_ndarray(loaded_array1, loaded_array2, save_name,  expansion_rate=expansion_rate, image_sequence1_rate=image1_rate, contrast_rate=contrast_rate, save_type=save_type, frame_length=frame_length, loop=loop, max_color1=max_color1, min_color1=min_color1, max_color2=max_color2, min_color2=min_color2)
def make_synthesis2_from_npz(saliency_map, screen, position, save_name, image1_rate=0.5, contrast_rate=1, save_type='gif', frame_length=160, loop=0, max_color1=[0,0,0], min_color1=[255,255,255], max_color2=[0,0,0], min_color2=[255,255,255]):
    saliency_map = load_ndarray(saliency_map)
    screen = load_ndarray(screen)
    position = load_ndarray(position)
    print('   saving '+save_name+'.gif...')
    blend_save_movie_from_ndarray_different_size(saliency_map, screen, position, save_name, image_sequence1_rate=image1_rate, contrast_rate=contrast_rate, save_type=save_type, frame_length=frame_length, loop=loop, max_color1=max_color1, min_color1=min_color1, max_color2=max_color2, min_color2=min_color2)

def save_movies_from_npzs(npz_name, episode_number_ndarray, reward_ndarray, save_name, expansion_rate=1, frame_length=160):
    global processing_file_number
    print(' saving '+save_name+'...')
    saved_episode = load_ndarray(episode_number_ndarray)
    reward_ndarray = load_ndarray(reward_ndarray)
    #print(saved_episode)
    digit = len(str(saved_episode[-1]))
    #print(digit)
    for i in count():
        if os.path.exists('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/files/'+npz_name+f'{i:0=3}'+'.npz')==False:
            break
        episode_num = str(saved_episode[i]).rjust(digit,'0')
        #print('  No.'+f'{i:0=3}'+'/'+str(number_of_files-1)+'...')
        print('  No.'+str(processing_file_number)+'/'+str(number_of_files))
        processing_file_number += 1
        save_movie_from_npz(npz_name+f'{i:0=3}',save_name+' ,ep'+episode_num+' ,ave_rew'+f'{reward_ndarray[i]:.2f}', expansion_rate=expansion_rate,frame_length=frame_length)
def blend_save_movies_from_npzs(npz_name1, npz_name2, episode_number_ndarray, reward_ndarray, save_name, expansion_rate=1,  image1_rate=0.5, contrast_rate=1, frame_length=160, loop=0, max_color1=[0,0,0], min_color1=[255,255,255], max_color2=[0,0,0], min_color2=[255,255,255]):
    global processing_file_number
    print(' saving '+save_name+'...')
    saved_episode = load_ndarray(episode_number_ndarray)
    reward_ndarray = load_ndarray(reward_ndarray)
    #print(saved_episode)
    digit = len(str(saved_episode[-1]))
    #print(digit)
    for i in count():
        if os.path.exists('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/files/'+npz_name1+f'{i:0=3}'+'.npz')==False:
            break
        episode_num = str(saved_episode[i]).rjust(digit,'0')
        #print(episode_num)
        #print('  No.'+f'{i:0=3}'+'/'+str(number_of_files-1)+'...')
        print('  No.'+str(processing_file_number)+'/'+str(number_of_files))
        processing_file_number += 1
        blend_save_movie_from_npz(npz_name1+f'{i:0=3}', npz_name2+f'{i:0=3}', save_name+' ,ep'+episode_num+' ,ave_rew'+f'{reward_ndarray[i]:.2f}', expansion_rate=expansion_rate,  image1_rate=image1_rate, contrast_rate=contrast_rate, frame_length=frame_length, loop=loop, max_color1=max_color1, min_color1=min_color1, max_color2=max_color2, min_color2=min_color2)
def make_synthesis2s_from_npzs(saliency_map, screen, position, episode_number_ndarray, reward_ndarray, save_name, image1_rate=0.5, contrast_rate=1, frame_length=160, loop=0, max_color1=[0,0,0], min_color1=[255,255,255], max_color2=[0,0,0], min_color2=[255,255,255]):
    global processing_file_number
    print(' saving '+save_name+'...')
    saved_episode = load_ndarray(episode_number_ndarray)
    reward_ndarray = load_ndarray(reward_ndarray)
    digit = len(str(saved_episode[-1]))
    #print(digit)
    for i in count():
        if os.path.exists('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/files/'+saliency_map+f'{i:0=3}'+'.npz')==False:
            break
        episode_num = str(saved_episode[i]).rjust(digit,'0')
        #print(episode_num)
        #print('  No.'+f'{i:0=3}'+'/'+str(number_of_files-1)+'...')
        print('  No.'+str(processing_file_number)+'/'+str(number_of_files))
        processing_file_number += 1
        make_synthesis2_from_npz(saliency_map+f'{i:0=3}', screen+f'{i:0=3}', position+f'{i:0=3}', save_name+' ,ep'+episode_num+' ,ave_rew'+f'{reward_ndarray[i]:.2f}', image1_rate=image1_rate, contrast_rate=contrast_rate, frame_length=frame_length, loop=loop, max_color1=max_color1, min_color1=min_color1, max_color2=max_color2, min_color2=min_color2)

def get_number_of_files():
    saved_episode = load_ndarray('episodes')
    if SAVE_SCREEN==True and variables2["SAVE_SCREEN"]==True:
        return len(saved_episode)*4
    else:
        return len(saved_episode)*2


if RESULT_NUMBER == 0:
    for i in count():
        if os.path.exists('results/result'+str(i+1))==False:
            for j in count():
                #print(i-j)
                if i == j:
                    print(' There are no folders in which gifs were not generated')
                    sys.exit()
                if os.path.exists('results/result'+str(i-j)+'/processed')==False:
                    RESULT_NUMBER = i-j
                    break
            break

print(' in result'+str(RESULT_NUMBER)+'...')

if os.path.exists('results/result'+str(RESULT_NUMBER)+'/processed')==True:
    print(' gifs were already generated in this folder')
    sys.exit()

with open('results/result'+str(RESULT_NUMBER)+'/files/variables.pickle', mode='rb') as f:
    variables=pickle.load(f)

for epi_num in count():
    if epi_num == variables["EPISODE_NUMBER"]:
        break
    if os.path.exists('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num))==False:
        continue

    print(' in episode'+str(epi_num)+'...')


    with open('results/result'+str(RESULT_NUMBER)+'/files/episode'+str(epi_num)+'/files/variables.pickle', mode='rb') as f:
        variables2=pickle.load(f)

    number_of_files = get_number_of_files()
    processing_file_number = 1
    save_movies_from_npzs('input','episodes','rewards','input',expansion_rate=8,frame_length=FRAME_LENGTH)
    blend_save_movies_from_npzs('saliency_map','input','episodes','rewards','synthesis', expansion_rate=8, loop=0, max_color1=SALIENCY_MAX_COLOR, min_color1=SALIENCY_MIN_COLOR, image1_rate=SALIENCY_MAP_RATE, contrast_rate=CONTRAST_MAGNIFICATION,frame_length=FRAME_LENGTH)
    if SAVE_SCREEN==True and variables2["SAVE_SCREEN"]==True:
        save_movies_from_npzs('screen','episodes','rewards','screen',expansion_rate=1,frame_length=FRAME_LENGTH)
        make_synthesis2s_from_npzs('saliency_map','screen','cart_location','episodes','rewards','synthesis2', loop=0, max_color1=SALIENCY_MAX_COLOR, min_color1=SALIENCY_MIN_COLOR, image1_rate=SALIENCY_MAP_RATE, contrast_rate=CONTRAST_MAGNIFICATION,frame_length=FRAME_LENGTH)

f = open('results/result'+str(RESULT_NUMBER)+'/processed',mode='w')
f.close()

print('Complete')
