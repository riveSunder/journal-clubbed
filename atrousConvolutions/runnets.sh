python multiscaleNet.py --model='UNet' --maxSteps=5000 --dispIt=100 --batchSize=8 --lR=9e-6 --restore=False

python multiscaleNet.py --model='MDAC' --maxSteps=5000 --dispIt=100 --batchSize=8 --lR=9e-6 --restore=False

python simplyFullyCNNet.py --maxSteps=5000 --dispIt=100 --batchSize=8 --lR=9e-6 --restore=False

#python atrousNet.py --restore=False --useAtrous=True --useUNet=False --maxSteps=1200 --dispIt=20 --lR=3e-4 --poolStride=4

#python atrousNet.py --restore=False --useAtrous=False --useUNet=False --maxSteps=1200 --dispIt=20 --lR=3e-4 --poolStride=4

#python atrousNet.py --restore=False --useAtrous=False --useUNet=True --maxSteps=1200 --dispIt=20 --lR=3e-4 --poolStride=4

#python atrousNet.py --restore=False --useAtrous=True --useUNet=True --maxSteps=1200 --dispIt=20 --lR=3e-4 --poolStride=4

