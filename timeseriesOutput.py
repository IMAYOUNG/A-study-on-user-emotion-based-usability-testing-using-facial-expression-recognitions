import pandas as pd
import matplotlib.pyplot as plt

def plot(filepath):
    df = pd.read_csv(filepath, skipfooter=5, engine='python')
    df = df.iloc[:-5]

    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')

    plt.figure(figsize=(20, 8))

    emotions = ['angry', 'fear', 'sad', 'disgust', 'neutral', 'surprise', 'happy']
    color_mapping = {
        'angry': 'red',
        'fear': 'deepPink',
        'sad': 'purple',
        'disgust' : 'brown',
        'neutral': 'gray',
        'happy': 'green',
        'surprise': 'yellow'
    }  

    for idx, emotion in enumerate(emotions, start=1):
        emotion_df = df[df['Predicted Emotion'] == emotion]
        plt.plot(emotion_df['Time'], [idx]*len(emotion_df), 'o', color=color_mapping[emotion], label=emotion, markersize=11)

    plt.yticks(range(1, len(emotions)+1), emotions, fontsize=34)  # y축 글씨 크기 조정
    plt.xticks(fontsize=20)
    plt.xlabel('Time', fontsize=27)  # x축 글씨 크기 조정
    plt.ylabel('Emotion', fontsize=27)  # y축 글씨 크기 조정
    plt.title('Emotion Time', fontsize=27)  # 제목 글씨 크기 조정
    
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
