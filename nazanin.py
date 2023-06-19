import gym
import random
import numpy as np
import time
from math import e
######################################## set environment and variables
                                 #tozihat                  #dar daryache sor nakhore  #age ansi bashe visual nemibinim
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode='ansi')
action_size= env.action_space.n # در این بازی تعداد اکشن هاشو نشون میده این متد
state_size = env.observation_space.n 
qtable = np.zeros((state_size,action_size))#اندازه ماتریس کیو رو نشان میدیم اول سطر دوم ستون

print("Init Qtable:\n ")
print(qtable)
total_episodes = 250 #تعداد کل اپیزود هایی که میخواهیم داشته باشیم

max_steps = 20 #تعداد حرکت در هر اپیزود
gamma = 0.8
epsilon=1


######################################### Game is on

print("training...")
for episode in range(total_episodes):
    state = env.reset()[0] #در هر بار شروع باید صفر بشه state
    step= 0 #step هم باید صفر شود
    #اگر به هدف برسه یا در دریاچه بیفته done میشه true پس در هربار شروع حلقه باید بشه false
    done = False
    print("EPISODE: ", episode)
    for step in range(max_steps):
        #random action
        random1 = np.random.random()
        if (random1<epsilon):
            action = env.action_space.sample()           
        if (random1>epsilon):
            action = np.argmax(qtable[state,:])
            
        action= env.action_space.sample() #یکی از اگشن هارو برمیداره یا بالا یا پایین یا چپ یا راست که به جای هرکدوم از اینا 0 یا 1 یا 2 یا 3 برمیگردونه
        env.render() #هربار بخوایم env رو به روزرسانی کنیم
        new_state, reward , done , truncated ,info = env.step(action)
        #وقتی میرسه به step جدید 3 چیز مهمه: خونه ی جدید. امتیازش و اینکه info چی هست
        
        if done and reward == 0: #اگه done=true بود ولی جایزه ای نگرفته بود یعنی تو چاله افتاده
            reward = -5
        if new_state == state: #اگه حرکتی نکردی فقط درجا زدی
            reward = -1
        print("NEW STATE: ", new_state,"REWARD: ",reward)
        qtable[state, action]= reward + gamma * np.max(qtable[new_state, :])
        print("QTABLE AT", state,qtable[state])
        state = new_state
        if done:
            print("GAME OVER.\n\n")
    print("new QTABLE")
    print(qtable)

env.reset()
env.close()
        
        
####################################### evaluation
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode='human')        
state = env.reset()[0]
step=0
done=False
for step in range(max_steps):
    env.render()#محیط گرافیکی نمایش داده بشه
    action= np.argmax(qtable[state,:])#اندیس ستون خونه ای بیشترین امتیاز داره رو برمیگردونه
    new_state, reward, done, truncated ,info = env.step(action)# مقصد بعدی میشه عدد action
    if done: #موقعی true میشه یا به مقصد میرسه یا تو چاله
        break
    state=new_state
        
time.sleep(2)
env.reset()
env.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        