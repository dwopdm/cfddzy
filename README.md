cfdfinal.py中TVD使用限制器minmod和superbee，FDS使用Roe， FVS使用了VanLeer，WENO3,GVC.finalgif版实现了动图输出，这些版本使用的ssp-rk3,在cfdrk3.py使用的标准版，使用ai添加了中文注释
总结来说，FDS方法比FVS精度更高，fds下tvd>weno3>gvc,fvs下weno3>tvd>gvc
exact.py实现了精确解在任何给定条件下的求解
![image](https://github.com/user-attachments/assets/f7576d86-054a-467e-b1bc-cf722edf7a63)

结果在final图片中，动图结果在sod_tube_anim_T0.25_MULTI_NX200.gif
