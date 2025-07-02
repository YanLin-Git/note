# docker

## 在docker中部署n8n服务
1. 安装docker
2. 创建docker volume `docker volume create n8n_data`
    > 通过使用卷，数据可以在容器停止或者删除时，仍然保留
3. 拉取镜像 `docker pull n8nio/n8n`
4. 创建并启动容器
    ```shell
        docker run -itd \
        --name n8n \
        -p 5678:5678 \
        -v n8n_data:/home/node/.n8n \
        -e N8N_SECURE_COOKIE=false \
        n8nio/n8n
    ```

    <details>
    <summary><b>参数介绍</b></summary>

    - -i: 交互式运行
    - -t: 分配一个伪终端
    - -d: 服务在后台运行
    - --name: 指定容器名称
    - -p: 主机端口:容器端口
    - -v: 挂载卷:容器目录
    - -e: 设置环境变量

    </details>
5. 其他命令
    1. 查看所有容器 `docker ps -a`
    2. 启动容器 `docker start 容器id`
    3. 停止容器 `docker stop 容器id`
    4. 重启容器 `docker restart 容器id`
    5. 删除容器 `docker rm 容器id`
    6. 进入容器 `docker exec -it 容器id /bin/bash`

