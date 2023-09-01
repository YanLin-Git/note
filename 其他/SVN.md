# SVN

## 一、安装
### 1.1、mac版安装
1. 查看是否安装 `brew list`
2. 安装 `brew install subversion`
3. 查看版本 `svn --version`
### 1.2、windows安装
- 参考 https://www.runoob.com/svn/svn-install.html


## 二、新建自己的分支，开发
1. 新建分支(copy) `svn copy -m 'myname dev' trunk地址 自己的branch`
2. 拉取branch代码(checkout) `svn checkout 自己的branch地址 本地目录`
3. 代码开发
4. 提交，先更新一下: `svn update`，然后提交: `svn commit -m '修改日志'`

## 三、将branch代码合并到trunk
1. 拉取trunk代码 `svn checkout trunk地址 本地trunk目录`
2. 进入本地trunk目录 `cd 本地trunk目录`
3. 合并 `svn merge 自己的branch地址`
4. 最后commit `svn commit -m '修改日志'`