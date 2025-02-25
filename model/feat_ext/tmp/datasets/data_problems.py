from pathlib import Path
import pandas as pd
CSV_PATH = Path(__file__).parent / 'record_20241223-18_48_58-_out' / 'dataset.csv'
df = pd.read_csv(CSV_PATH)
move_counts = df['move'].value_counts()
attack_counts = df['attack'].value_counts()
skill_counts = df['skill'].value_counts()
weapon_counts = df['weapon'].value_counts()

# 打印统计结果
print("Move counts:")
print(move_counts)
print("Attack counts:")
print(attack_counts)
print("\nSkill counts:")
print(skill_counts)
print("\nWeapon counts:")
print(weapon_counts)
