
XCOPY *.py C:\datas /Y

c:
cd C:\datas


@rem python forest_UAV_U_Net_pytorch.py cedar 100 24 256 org_crop4Corner_5120_3584_Size0512_lap0256
@rem python forest_UAV_U_Net_pytorch.py cedar 100 24 256 org_crop4Corner_5376_3584_Size0256_lap0000
@REM python forest_UAV_U_Net_pytorch.py cypress 150 16 256 org_crop4Corner_5120_3584_Size0512_lap0256_rotate_flipMirror
@REM python forest_UAV_U_Net_pytorch.py cypress 30 64 256 org_crop4Corner_5120_3072_Size1024_lap0512_rotate_flipMirror OOME

@REM python forest_UAV_U_Net_pytorch.py cedar 150 32 256 org_crop4Corner_5120_3072_Size1024_lap0512_rotate_flipMirror
@REM python forest_UAV_U_Net_pytorch.py cypress 150 32 256 org_crop4Corner_5120_3072_Size1024_lap0512_rotate_flipMirror


python forest_UAV_U_Net_pytorch.py cypress 80 24 256 org_crop4Corner_5120_3584_Size0512_lap0256_rotate_flipMirror
python forest_UAV_U_Net_pytorch.py cedar 80 24 256 org_crop4Corner_5120_3584_Size0512_lap0256_rotate_flipMirror

python forest_UAV_U_Net_pytorch.py cypress 80 24 256 org_crop4Corner_5376_3584_Size0256_lap0000_rotate_flipMirror
python forest_UAV_U_Net_pytorch.py cedar 80 24 256 org_crop4Corner_5376_3584_Size0256_lap0000_rotate_flipMirror

h:
@REM cd H:\�}�C�h���C�u\Forest\src

