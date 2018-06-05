
python atrousNet.py --restore=False --useAtrous=True --useUNet=False --maxSteps=500 --dispIt=20 --lR=3e-5 --poolStride=4

python atrousNet.py --restore=False --useAtrous=False --useUNet=False --maxSteps=800 --dispIt=20 --lR=3e-5 --poolStride=4

python atrousNet.py --restore=False --useAtrous=False --useUNet=True --maxSteps=800 --dispIt=20 --lR=3e-5 --poolStride=4

python atrousNet.py --restore=False --useAtrous=True --useUNet=True --maxSteps=800 --dispIt=20 --lR=3e-5 --poolStride=4

