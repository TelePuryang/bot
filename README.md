


## 在国外vps能够正常构建并使用。国内网络环境有点问题
## 安装
1. 获取[OpenAI API](https://openai.com/api/) 

2. 获取电报机器人api [@BotFather](https://t.me/BotFather)
    创建机器人
    设置名称
    得到api
    
3. 在config目录下配置config.yml文件，配置openai的api和电报机器人的api

> 如果需要配置mogo的用户名和密码，进入config文件夹下编辑config.env文件，
通过http://服务器ip地址:8888端口 访问mongo数据库的后台。

[!done] 在这里存储了不同用户的数据。


## 通过Docker构建镜像
    ```bash
    docker-compose --env-file config/config.env up --build
    ```


