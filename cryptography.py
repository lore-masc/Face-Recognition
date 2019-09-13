import fernet
import argparse

def main(save=True,
         encrypt=True):
    if save is True:
        key = fernet.Fernet.generate_key()
        file = open('key\key.key', 'wb')
        file.write(key)
        file.close()
    else:
        file = open('key\key.key', 'rb')
        key = file.read()
        file.close()
    if encrypt:
        # Encrypt the file
        with open(FILE_PATH, 'rb') as f:
            data = f.read()

        f = fernet.Fernet(key)
        encrypted = f.encrypt(data)

        # Write the encrypted file
        with open(FILE_PATH+".encrypted", 'wb') as f:
            f.write(encrypted)
    else:
        # Decode the file
        with open(FILE_PATH+".encrypted", "rb") as f:
            image = f.read()
        f = fernet.Fernet(key)
        decrypted = f.decrypt(image)
        # Write the decrypted file
        with open(FILE_PATH, 'wb') as f:
            f.write(decrypted)


parser = argparse.ArgumentParser(description='Crypt and decrypt files')
parser.add_argument('-s', '--save', action='store_true', dest='save', help='set option in order to save new key')
parser.add_argument('-e', '--encrypt', action='store_true', dest='encrypt', help='set option in order to encrypt the file')
parser.add_argument('-f', '--file', action='store', dest='file', help='write the file relative path', required=True)
args = parser.parse_args()

FILE_PATH = args.file
main(save=args.save, encrypt=args.encrypt)






