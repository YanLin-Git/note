# Git学习笔记
## 一、运行git前的配置
1. git配置文件的3个级别
    1. 系统级 `/etc/gitconfig`
    2. 当前登录用户 `~/gitconfig`
    3. 当前git仓库 `.git/config`
2. 查看所有的配置，及它们所在的文件  
`git config --list --show-origin`
3. **安装git后，设置用户名和邮箱**
```shell
git config --global user.name 'ylnchen'
git config --global user.email 'ylnchen@tencent.com'
#也可以直接在配置文件中修改
```

## 二、git基础
1. 获取远程仓库 `git clone <url> 本地文件夹`
2. 查看哪些文件，处于什么状态 `git status` or `git status -s(--short)`
3. **添加新文件** (工作区--->暂存区) `git add file_new`
4. **文件修改后** (工作区--->暂存区) `git add file_modified`
5. 忽略某些文件，不需要跟踪。查看`.gitignore`文件，例如:

```shell
cat .gitignore

#忽略所有.o或.a文件
*.[oa]
# 但跟踪所有的 lib.a，即便你在前面忽略了 .a 文件
!lib.a

# 只忽略当前目录下的 TODO 文件，而不忽略 subdir/TODO
/TODO

# 忽略任何目录下名为 build 的文件夹
build/

# 忽略 doc/notes.txt，但不忽略 doc/server/arch.txt
doc/*.txt

# 忽略 doc/ 目录及其所有子目录下的 .pdf 文件
doc/**/*.pdf
```

6. 查看**工作区**与**暂存区**的差异 `git diff`
7. 查看**暂存区**与**仓库**的差异 `git diff --staged(--cached)`
8. **提交** (暂存区--->仓库) `git commit -m '备注信息'`
9. 移除文件
    1. 第一种方式:
    ```shell
    # 1. 工作区中删除，正常linux操作
    rm -f file_delete
    # 2. 暂存区中删除
    git rm file_delete
    # 3. 之后提交时，就不会再跟踪该文件
    ```
    2. 第二种方式：
    仅删除暂存区中的文件 `git rm --cached file_delete`
10. 移动文件 `git mv file_from file_to`  
    其实相当于
    ```shell
    mv file_from file_to
    git rm file_from
    git add file_to
    ```
11. 查看提交历史 `git log`
    1. **-p(--patch)** 查看具体提交内容
    2. **-2** 只查看最近两次提交
12. 重新提交
    第一次提交后，发现忘记添加一个文件，或者修正笔误等，可使用 `git commit --amend` 这样第二次提交会代替第一次提交
13. 撤销对文件对修改 (仓库--->工作区) `git checkout --<file>`

## 三、打标签
1. 列出已有标签 `git tag`  
    列出特定标签 `git tag -l 'v1.1.5*'`
2. 查看标签信息 `git show <标签名>`
3. 创建标签 `git tag -a <标签名> -m '备注信息'`
4. 对过去的提交打标签

```shell
#1. 查看提交历史
git log --pretty=oneline
#2. 打标签
git tag -a <标签名> 特定提交记录的hash值
```

5. 将标签推送到远程库 `git push <远程库> <标签名>`
6. 删除本地库中的轻量标签 `git tag -d v1.1.5-lw`
7. 删除远程库中的标签 `git push <远程库> --delete <标签名>`

## 四、远程仓库
1. 查看远程仓库 `git remote -v`
2. 从远程仓库拉取

```
git fetch <远程库>
git pull
```

3. 推送到远程 `git push <远程库> <本地库>`
4. 查看某个远程仓库 `git remote show <远程库>`
5. 远程仓库的重命名 `git remote rename old_name new_name`
6. 远程仓库的移除 `git remote remove <远程库>`
7. 删除远程仓库中的分支 `git push origin --delete <远程分支>`

## 五、分支
1. 创建分支 `git branch <branch>`
2. 切换到已存在的分支 `git checkout <branch>`
3. 创建分支并切换过去 `git checkout -b <branch>`
4. 合并分支 `git merge <branch>`
5. 删除分支 `git branch -d <branch>`
6. 查看本地所有分支 `git branch -v`
7. 查看所有分支 `git branch -a`
8. 拉取远程分支到本地 `git checkout -b <本地分支> origin/远程分支`