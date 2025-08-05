
# openvpn

## 一、服务端安装

1. 安装`openvpn`、`easy-rsa`
    - `sudo apt install openvpn easy-rsa`

2. 在`~/easy-rsa/vars`文件中设置环境变量
    
    <details>
    <summary>示例</summary>

    ```
    set_var EASYRSA_REQ_COUNTRY "CN"
    set_var EASYRSA_REQ_PROVINCE "BEIJING"
    set_var EASYRSA_REQ_CITY "BEIJING"
    set_var EASYRSA_REQ_ORG "test"
    # set_var EASYRSA_REQ_EMAIL "me@example.net"
    # set_var EASYRSA_REQ_OU "My Organizational Unit"
    ```

    </details>

3. 使用`easy-rsa`生成一些密钥、证书
    ```shell
    # 初始化pki (Public Key Infrastructure)
    sudo ./easyrsa init-pki 

    # 生成ca证书
    # 生成 pki/ca.crt 和 pki/private/ca.key
    sudo ./easyrsa build-ca nopass 

    # 生成dh证书
    # 生成pki/dh.pem
    sudo ./easyrsa gen-dh

    # 生成服务端密钥对
    # 生成 pki/issued/vpn-server.crt 和 pki/private/vpn-server.key
    sudo ./easyrsa build-server-full vpn-server nopass 

    # 生成客户端密钥对
    # 生成 pki/issued/vpn-client.crt 和 pki/private/vpn-client.key
    sudo ./easyrsa build-client-full vpn-client nopass
    ```

4. 使用`openvpn`生成静态密钥文件`ta.key`
    - `openvpn --genkey --secret pki/ta.key`

5. 将服务端所需的配置文件copy至`openvpn`的配置目录`/etc/openvpn/server`
    ```shell
    sudo cp pki/ca.crt /etc/openvpn/server/ca.crt 
    sudo cp pki/dh.pem /etc/openvpn/server/dh.pem 
    sudo cp pki/issued/vpn-server.crt /etc/openvpn/server/server.crt
    sudo cp pki/private/vpn-server.key /etc/openvpn/server/server.key 
    sudo cp pki/ta.key /etc/openvpn/server/ta.key
    ```

6. 在配置目录`/etc/openvpn/server`中，添加一个验证用户名密码的脚本`check.sh`
    <details>
    <summary>check.sh示例</summary>

    ```shell
    #!/bin/bash

    # 密码文件 用户名 密码明文
    PASSFILE="/etc/openvpn/openvpnfile" 
    # 用户登录情况的日志
    LOG_FILE="/var/log/openvpn-password.log"  

    TIME_STAMP=`date "+%Y-%m-%d %T"`
    if [ ! -r "${PASSFILE}" ]; then
        echo "${TIME_STAMP}: Could not open password file \"${PASSFILE}\" for reading." >> ${LOG_FILE}
        exit 1
    fi
    CORRECT_PASSWORD=`awk '!/^;/&&!/^#/&&$1=="'${username}'"{print $2;exit}'    ${PASSFILE}`
    if [ "${CORRECT_PASSWORD}" = "" ]; then
        echo "${TIME_STAMP}: User does not exist: username=\"${username}\",password=\"${password}\"." >> ${LOG_FILE}
        exit 1
    fi
    if [ "${password}" = "${CORRECT_PASSWORD}" ]; then
        echo "${TIME_STAMP}: Successful authentication: username=\"${username}\"." >> ${LOG_FILE}
        exit 0
    fi
    echo "${TIME_STAMP}: Incorrect password: username=\"${username}\", password=\"${password}\"." >> ${LOG_FILE}
    exit 1
    ```

    </details>

7. 该脚本会去验证用户输入的账号信息，是否与`/etc/openvpn/openvpnfile`中一致，我们需要创建一个`openvpnfile`
    <details>
    <summary>openvpnfile文件示例</summary>

    ```shell
    # 用户名:test 密码:test
    test test
    # 用户名:user 密码:admin
    user admin
    ```

    </details>

8. 前面准备工作都做完后，创建一个配置文件`/etc/openvpn/server/server.conf`，让openvpn能够找到这些信息，就可以启动服务了
    <details>
    <summary>server.conf文件示例</summary>

    ```conf
    #################################################
    # Sample OpenVPN 2.0 config file for            #
    # multi-client server.                          #
    #                                               #
    # This file is for the server side              #
    # of a many-clients <-> one-server              #
    # OpenVPN configuration.                        #
    #                                               #
    # OpenVPN also supports                         #
    # single-machine <-> single-machine             #
    # configurations (See the Examples page         #
    # on the web site for more info).               #
    #                                               #
    # This config should work on Windows            #
    # or Linux/BSD systems.  Remember on            #
    # Windows to quote pathnames and use            #
    # double backslashes, e.g.:                     #
    # "C:\\Program Files\\OpenVPN\\config\\foo.key" #
    #                                               #
    # Comments are preceded with '#' or ';'         #
    #################################################

    # Which local IP address should OpenVPN
    # listen on? (optional)
    ;local a.b.c.d

    # Which TCP/UDP port should OpenVPN listen on?
    # If you want to run multiple OpenVPN instances
    # on the same machine, use a different port
    # number for each one.  You will need to
    # open up this port on your firewall.
    port 1194

    # TCP or UDP server?
    ;proto tcp
    proto udp

    # "dev tun" will create a routed IP tunnel,
    # "dev tap" will create an ethernet tunnel.
    # Use "dev tap0" if you are ethernet bridging
    # and have precreated a tap0 virtual interface
    # and bridged it with your ethernet interface.
    # If you want to control access policies
    # over the VPN, you must create firewall
    # rules for the the TUN/TAP interface.
    # On non-Windows systems, you can give
    # an explicit unit number, such as tun0.
    # On Windows, use "dev-node" for this.
    # On most systems, the VPN will not function
    # unless you partially or fully disable
    # the firewall for the TUN/TAP interface.
    ;dev tap
    dev tun

    # Windows needs the TAP-Win32 adapter name
    # from the Network Connections panel if you
    # have more than one.  On XP SP2 or higher,
    # you may need to selectively disable the
    # Windows firewall for the TAP adapter.
    # Non-Windows systems usually don't need this.
    ;dev-node MyTap

    # SSL/TLS root certificate (ca), certificate
    # (cert), and private key (key).  Each client
    # and the server must have their own cert and
    # key file.  The server and all clients will
    # use the same ca file.
    #
    # See the "easy-rsa" directory for a series
    # of scripts for generating RSA certificates
    # and private keys.  Remember to use
    # a unique Common Name for the server
    # and each of the client certificates.
    #
    # Any X509 key management system can be used.
    # OpenVPN can also use a PKCS #12 formatted key file
    # (see "pkcs12" directive in man page).
    ca ca.crt
    cert server.crt
    key server.key  # This file should be kept secret

    # Diffie hellman parameters.
    # Generate your own with:
    #   openssl dhparam -out dh2048.pem 2048
    ;dh dh2048.pem
    dh dh.pem

    # Network topology
    # Should be subnet (addressing via IP)
    # unless Windows clients v2.0.9 and lower have to
    # be supported (then net30, i.e. a /30 per client)
    # Defaults to net30 (not recommended)
    ;topology subnet

    # Configure server mode and supply a VPN subnet
    # for OpenVPN to draw client addresses from.
    # The server will take 10.8.0.1 for itself,
    # the rest will be made available to clients.
    # Each client will be able to reach the server
    # on 10.8.0.1. Comment this line out if you are
    # ethernet bridging. See the man page for more info.
    server 10.8.0.0 255.255.255.0

    # Maintain a record of client <-> virtual IP address
    # associations in this file.  If OpenVPN goes down or
    # is restarted, reconnecting clients can be assigned
    # the same virtual IP address from the pool that was
    # previously assigned.
    ifconfig-pool-persist ipp.txt

    # Configure server mode for ethernet bridging.
    # You must first use your OS's bridging capability
    # to bridge the TAP interface with the ethernet
    # NIC interface.  Then you must manually set the
    # IP/netmask on the bridge interface, here we
    # assume 10.8.0.4/255.255.255.0.  Finally we
    # must set aside an IP range in this subnet
    # (start=10.8.0.50 end=10.8.0.100) to allocate
    # to connecting clients.  Leave this line commented
    # out unless you are ethernet bridging.
    ;server-bridge 10.8.0.4 255.255.255.0 10.8.0.50 10.8.0.100

    # Configure server mode for ethernet bridging
    # using a DHCP-proxy, where clients talk
    # to the OpenVPN server-side DHCP server
    # to receive their IP address allocation
    # and DNS server addresses.  You must first use
    # your OS's bridging capability to bridge the TAP
    # interface with the ethernet NIC interface.
    # Note: this mode only works on clients (such as
    # Windows), where the client-side TAP adapter is
    # bound to a DHCP client.
    ;server-bridge

    # Push routes to the client to allow it
    # to reach other private subnets behind
    # the server.  Remember that these
    # private subnets will also need
    # to know to route the OpenVPN client
    # address pool (10.8.0.0/255.255.255.0)
    # back to the OpenVPN server.
    ;push "route 192.168.10.0 255.255.255.0"
    ;push "route 192.168.20.0 255.255.255.0"

    # To assign specific IP addresses to specific
    # clients or if a connecting client has a private
    # subnet behind it that should also have VPN access,
    # use the subdirectory "ccd" for client-specific
    # configuration files (see man page for more info).

    # EXAMPLE: Suppose the client
    # having the certificate common name "Thelonious"
    # also has a small subnet behind his connecting
    # machine, such as 192.168.40.128/255.255.255.248.
    # First, uncomment out these lines:
    ;client-config-dir ccd
    ;route 192.168.40.128 255.255.255.248
    # Then create a file ccd/Thelonious with this line:
    #   iroute 192.168.40.128 255.255.255.248
    # This will allow Thelonious' private subnet to
    # access the VPN.  This example will only work
    # if you are routing, not bridging, i.e. you are
    # using "dev tun" and "server" directives.

    # EXAMPLE: Suppose you want to give
    # Thelonious a fixed VPN IP address of 10.9.0.1.
    # First uncomment out these lines:
    ;client-config-dir ccd
    ;route 10.9.0.0 255.255.255.252
    # Then add this line to ccd/Thelonious:
    #   ifconfig-push 10.9.0.1 10.9.0.2

    # Suppose that you want to enable different
    # firewall access policies for different groups
    # of clients.  There are two methods:
    # (1) Run multiple OpenVPN daemons, one for each
    #     group, and firewall the TUN/TAP interface
    #     for each group/daemon appropriately.
    # (2) (Advanced) Create a script to dynamically
    #     modify the firewall in response to access
    #     from different clients.  See man
    #     page for more info on learn-address script.
    ;learn-address ./script

    # If enabled, this directive will configure
    # all clients to redirect their default
    # network gateway through the VPN, causing
    # all IP traffic such as web browsing and
    # and DNS lookups to go through the VPN
    # (The OpenVPN server machine may need to NAT
    # or bridge the TUN/TAP interface to the internet
    # in order for this to work properly).
    ;push "redirect-gateway def1 bypass-dhcp"

    # Certain Windows-specific network settings
    # can be pushed to clients, such as DNS
    # or WINS server addresses.  CAVEAT:
    # http://openvpn.net/faq.html#dhcpcaveats
    # The addresses below refer to the public
    # DNS servers provided by opendns.com.
    ;push "dhcp-option DNS 208.67.222.222"
    ;push "dhcp-option DNS 208.67.220.220"

    # Uncomment this directive to allow different
    # clients to be able to "see" each other.
    # By default, clients will only see the server.
    # To force clients to only see the server, you
    # will also need to appropriately firewall the
    # server's TUN/TAP interface.
    client-to-client

    # Uncomment this directive if multiple clients
    # might connect with the same certificate/key
    # files or common names.  This is recommended
    # only for testing purposes.  For production use,
    # each client should have its own certificate/key
    # pair.
    #
    # IF YOU HAVE NOT GENERATED INDIVIDUAL
    # CERTIFICATE/KEY PAIRS FOR EACH CLIENT,
    # EACH HAVING ITS OWN UNIQUE "COMMON NAME",
    # UNCOMMENT THIS LINE OUT.
    ;duplicate-cn

    # The keepalive directive causes ping-like
    # messages to be sent back and forth over
    # the link so that each side knows when
    # the other side has gone down.
    # Ping every 10 seconds, assume that remote
    # peer is down if no ping received during
    # a 120 second time period.
    keepalive 10 120

    # For extra security beyond that provided
    # by SSL/TLS, create an "HMAC firewall"
    # to help block DoS attacks and UDP port flooding.
    #
    # Generate with:
    #   openvpn --genkey --secret ta.key
    #
    # The server and each client must have
    # a copy of this key.
    # The second parameter should be '0'
    # on the server and '1' on the clients.
    tls-auth ta.key 0 # This file is secret

    # Select a cryptographic cipher.
    # This config item must be copied to
    # the client config file as well.
    # Note that v2.4 client/server will automatically
    # negotiate AES-256-GCM in TLS mode.
    # See also the ncp-cipher option in the manpage
    cipher AES-256-CBC

    # Enable compression on the VPN link and push the
    # option to the client (v2.4+ only, for earlier
    # versions see below)
    ;compress lz4-v2
    ;push "compress lz4-v2"

    # For compression compatible with older clients use comp-lzo
    # If you enable it here, you must also
    # enable it in the client config file.
    ;comp-lzo

    # The maximum number of concurrently connected
    # clients we want to allow.
    ;max-clients 100

    # It's a good idea to reduce the OpenVPN
    # daemon's privileges after initialization.
    #
    # You can uncomment this out on
    # non-Windows systems.
    ;user nobody
    ;group nobody

    # The persist options will try to avoid
    # accessing certain resources on restart
    # that may no longer be accessible because
    # of the privilege downgrade.
    persist-key
    persist-tun

    # Output a short status file showing
    # current connections, truncated
    # and rewritten every minute.
    status openvpn-status.log

    # By default, log messages will go to the syslog (or
    # on Windows, if running as a service, they will go to
    # the "\Program Files\OpenVPN\log" directory).
    # Use log or log-append to override this default.
    # "log" will truncate the log file on OpenVPN startup,
    # while "log-append" will append to it.  Use one
    # or the other (but not both).
    ;log         openvpn.log
    ;log-append  openvpn.log

    # Set the appropriate level of log
    # file verbosity.
    #
    # 0 is silent, except for fatal errors
    # 4 is reasonable for general usage
    # 5 and 6 can help to debug connection problems
    # 9 is extremely verbose
    verb 3

    # Silence repeating messages.  At most 20
    # sequential messages of the same message
    # category will be output to the log.
    ;mute 20

    # Notify the client that when the server restarts so it
    # can automatically reconnect.
    explicit-exit-notify 1

    # 认证信息加密方式
    auth SHA1

    # 支持密码认证，用户密码登陆方式验证
    auth-user-pass-verify ./check.sh via-env
    auth-nocache
    username-as-common-name

    # 允许使用自定义脚本
    script-security 3

    # 没有客户端crt也能登录，我感觉不好，别开。
    ;client-cert-not-required

    ```

    </details>

9. 启动服务，并设置开机自启
    ```shell
    sudo systemctl start openvpn-server@server
    sudo systemctl enable openvpn-server@server
    ```
    > `openvpn-server@server`，会去`/etc/openvpn/server/`目录下，读取`server.conf`

## 二、派发到客户端的conf
> 至此，服务端已经启动了。为了让客户端能连接自己，还需要创建一个客户端的配置文件`client.conf`

<details>
<summary>client.conf文件示例</summary>

```conf
# client指明这是个客户端文件
client
# 验证远程服务器有没有开启tls，如果你没开，就把这个注释掉
remote-cert-tls server
# 失败无限重连
resolv-retry infinite

# tun模式
dev tun

# 公网IP, 端口，UDP/TCP
# remote x.x.x.x <port>
remote 192.168.103.213 1194
proto udp

# 不主动拉取服务端路径，防止全部流量都走VPN
route-nopull
# 限定10.8.0.0/24的流量经过VPN
route 10.8.0.0  255.255.255.0 vpn_gateway

# 加密方法和压缩方法
# 与server.conf中一致
cipher AES-256-CBC
auth SHA1
;comp-lzo

# 使用密码验证
auth-user-pass
# 不绑定固定端口，客户端都这样
nobind
# 允许循环路由，要不然找不到家
allow-recursive-routing

<ca>
-----BEGIN CERTIFICATE-----
# pki/ca.crt 中的内容
-----END CERTIFICATE-----
</ca>


<key>
-----BEGIN PRIVATE KEY-----
# pki/private/vpn-client.key 中的内容
-----END PRIVATE KEY-----
</key>


<cert>
-----BEGIN CERTIFICATE-----
# pki/issued/vpn-client.crt 中的内容
-----END CERTIFICATE-----
</cert>


key-direction 1
<tls-auth>
#
# 2048 bit OpenVPN static key
#
-----BEGIN OpenVPN Static key V1-----
# pki/ta.key 中的内容
-----END OpenVPN Static key V1-----
</tls-auth>
```

</details>


# 三、客户端安装

1. 安装openvpn
    - `sudo apt install openvpn`

2. 获取服务端派发的conf
    - 将上面第二节生成的client.conf，拷贝到`openvpn`客户端的配置目录`/etc/openvpn/client/`
    - 具体命令: `sudo cp path/to/your/conf /etc/openvpn/client/`

3. 启动服务，并设置开机自启
    ```shell
    sudo systemctl start openvpn-client@client
    sudo systemctl enable openvpn-client@client
    ```

    > `openvpn-client@client`，会去`/etc/openvpn/client/`目录下，读取`client.conf`



- 参考教程: https://zhuanlan.zhihu.com/p/3446441267
