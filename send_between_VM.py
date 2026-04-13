import subprocess
import os
import yaml

with open ('config.yml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
raw_data_path = config.get('folders').get('raw_data')
csv_mail_path = config.get('folders').get('csv_mail')
csv_calls_path = config.get('folders').get('csv_calls')
mail_filename = config.get('folders').get('mail_filename')
calls_filename = config.get('folders').get('calls_filename')
        
DEST_USER = "helen-markova"
DEST_DNS = 'compute-vm-4-8-50-ssd-1775042462973.ru-central1.internal'
DEST_PATH_MAIL = "/home/helen-markova/ask_me/csv_mail"
DEST_PATH_CALLS = "/home/helen-markova/ask_me/csv_calls"
SSH_KEY = "/home/helen-markova/.ssh/id_rsa_source"

def transfer_files(source_dir, dest_dir):
    cmd = [
        "rsync",
        "-avz",  # archive, verbose, compress
        "-e", f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=accept-new",
        source_dir,
        f"{DEST_USER}@{DEST_DNS}:{dest_dir}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Transfer successful!")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Transfer failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def transfer_all():
    transfer_files(os.path.join(csv_mail_path, mail_filename),  DEST_PATH_MAIL)
    transfer_files(os.path.join(csv_calls_path, calls_filename), DEST_PATH_CALLS)


if __name__ == "__main__":
    transfer_all()
