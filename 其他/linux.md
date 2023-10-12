# linux常用命令

## 1、ssh登录
- `ssh 用户名@服务器地址 -p port`

## 2、拷贝数据
1. `scp 源地址 目的地址`
    - 格式: `scp 用户名@服务器地址:文件所在目录 本地目录`
2. `rsync -avP 源地址 目的地址`
    - 格式: `rsync -avP 用户名@服务器地址:文件所在目录 本地目录`

## 3、建立软链接
`ln -s 源地址 目的地址`

## 4、解压缩
1. `tar -czvf 压缩包名称 待压缩文件1 待压缩文件2 ...`
2. `tar -xzvf 压缩包名称`
    - `-c`，压缩
    - `-x`，解压缩
    - `-z`，使用gzip压缩算法
    - `-v`，在屏幕上显示详细信息
    - `-f`，后面紧跟着 文件名，(-f参数一定要写在最后)

## 5、查看CPU个数
1. 有几个物理CPU `cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l`
2. 每个CPU中有几个核 `cat /proc/cpuinfo | grep "cpu cores" | uniq`
3. 有几个逻辑CPU `cat /proc/cpuinfo | grep "processor" | wc -l`