import os
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('solarport.stanford.edu', username="attie")
sftp = ssh.open_sftp()
localpath = '/Users/rattie/Data/SDO/HMI/export_temp/'
remotepath = '/opt/crestelsetup/patchzip/abc.txt'
sftp.put(localpath, remotepath)
sftp.close()
ssh.close()




 # Using paramko_scp
# def createSSHClient(server, port, user, password):
#     client = paramiko.SSHClient()
#     client.load_system_host_keys()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect(server, port, user, password)
#     return client
#
# ssh = createSSHClient(server, port, user, password)
# scp = SCPClient(ssh.get_transport())