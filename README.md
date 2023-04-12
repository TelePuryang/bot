<img src="hello.png" alt="#" width="300" >
## 在国外vps能够正常构建并使用。国内网络环境有点问题
## 安装
1. 获取[OpenAI API](https://openai.com/api/) 

2. 获取电报机器人api [@BotFather](https://t.me/BotFather)
    创建机器人
    设置名称
    得到api
    
3. 在config目录下配置config.yml文件，配置openai的api和电报机器人的api

##（可以进config.env文件设置数据库的访问用户和密码）
> 如果需要配置mogo的用户名和密码，进入config文件夹下编辑config.env文件，
通过http://服务器公网ip地址:8081端口 访问mongo数据库的后台。


## 如果需要关闭公网访问mongodb数据库，请在docker-compose.yml文件中修改port映射为主机本地地址；
如：
将
```d
     ports:
      - 127.0.0.1:${MONGO_EXPRESS_PORT:-8081}:${MONGO_EXPRESS_PORT:-8081}
```


 改为
    
```d
     ports:
      - ${MONGO_EXPRESS_PORT:-8081}:${MONGO_EXPRESS_PORT:-8081}
```



## 通过Docker-compose构建容器集群
```bash
    docker-compose --env-file config/config.env up --build
```


##完成！
