# Windows10 Hadoop Config Notes

This ia a note for install and config hadoop 3.2.2 on Windows 10.

## Install OpenSSH Server

1. Install sshd service on Windows 10(OpenSSH). See this [doc](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse) for any detail.
    ```powershell
    # Install the OpenSSH Client
    Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

    # Install the OpenSSH Server
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
    ```
2. Check status:
    ```powershell
    Get-WindowsCapability -Online | ? Name -like 'OpenSSH*'
    ```
3. Config and start server
    ```powershell
    # Start the sshd service
    Start-Service sshd

    # OPTIONAL but recommended:
    Set-Service -Name sshd -StartupType 'Automatic'

    # Confirm the firewall rule is configured. It should be created automatically by setup.
    Get-NetFirewallRule -Name *ssh*

    # There should be a firewall rule named "OpenSSH-Server-In-TCP", which should be enabled
    # If the firewall does not exist, create one
    New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
    ```
4. If you (the user) are in the administrator group, edit `C:\ProgramData\ssh\sshd_config` to comment the last 2 lines:
    ```config
    # Match Group administrators
    #       AuthorizedKeysFile __PROGRAMDATA__\ssh\administrators_authorized_keys
    ```
    This is because if you are in administrator, then you need a special `administrator_authorized_keys` in `C:\ProgramData\ssh\`, instead of a common `authorized_keys` in the user's `.ssh` directory like that in Linux.

## Install and Config Hadoop

Before continuing, I assume you have install **JDK1.8.0**. Then do:

### Install winutils for hadoop

**winutils** is necessary for running hadoop on windows. See this [repository](https://github.com/cdarlint/winutils) for any detail.

1. Download the package
2. copy into according bin directory.

### Install Hadoop

1. Download and untar the [hadoop binary](https://hadoop.apache.org/releases.html).
2. add `HADOOP_HOME` assigned with the directory to your enviroment variables.
3. add `%HADOOP_HOME%\bin` and `%HADOOP_HOME%\sbin` to your `%PATH%`
4. type in `hadoop` in terminal to see whether your config is right. (That means `hadoop` should be found and executed)

### Build Hadoop

#### Build Protobuf

1. Download `protoc-2.5.0-win32.zip` and `protobuf-2.5.0.zip`
 from [here](https://github.com/google/protobuf/releases/tag/v2.5.0)
2. unzip `protoc.exe` to `protobuf-2.5.0/src`, and add this directory into environment variables.
3. go into `protobuf-2.5.0/java` and run
    ```powershell
    mvn install
    ```

#### Build Hadoop Itself
Assume you have installed `maven` before this.

1. run this to compile and package the Hadoop project:
    ```powershell
    mvn package -Pdist,native-win -DskipTests -Dtar
    ```
2. a

### Config Hadoop and HDFS

We need to config some files in `%HADOOP_HOME%\etc\hadoop\`.

1. `core-site.xml`. This specified the server's port.
    ```xml
    <configuration>
        <property>
            <name>fs.defaultFS</name>
            <value>hdfs://localhost:9000</value>
        </property>
    </configuration>
    ```
2. `hdfs-site.xml`. Specify the directories for namenode's and datanode's data.
    ```xml
    <configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>F:\.hadoop\name</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>F:\.hadoop\data</value>
    </property>
    </configuration>
    ```
3. `mapred-site.xml` and `yarn-site.xml`. Do this if you want to run mapreduce job on **yarn**
    ```xml
    <!-- mapred-site.xml -->
    <configuration>
        <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
        </property>
        <property>
            <name>mapreduce.application.classpath</name>
            <value>$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/*:$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*</value>
        </property>
    </configuration>

    <!-- yarn-site.xml -->
    <configuration>
        <property>
            <name>yarn.nodemanager.aux-services</name>
            <value>mapreduce_shuffle</value>
        </property>
        <property>
            <name>yarn.nodemanager.env-whitelist</name>
            <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_MAPRED_HOME</value>
        </property>
    </configuration>
    ```
4. if you want to use jdk that is not in your system environment variables, you can specify it in `hadoop-env.cmd`. Just add:
    ```powershell
    set JAVA_HOME=%YOUR_PREFERED_JAVA_HOME%
    ```

### Initialize HDFS

1. run:
    ```powershell
    hdfs namenode -format
    ```
    then you will see
    ```powershell
    INFO common.Storage: Storage directory F:\.hadoop\name has been successfully formatted.
    ```
2. start up hdfs:
    ```powershell
    start-all
    ```
    you will see some cmd line windows pop up. If no error in those windows, then you've succeed!

### Test Hadoop

1. mkdir for further usage (**hank** is just my name, use your user name on the system is more convenient):
    ```powershell
    hdfs dfs -mkdir -p /user/hank
    ```
2. prepare data for test
    ```powershell
    hdfs dfs -put "$($env:HADOOP_HOME)\etc\hadoop\*.xml" input
    hdfs dfs -ls input
    ```
    If this proceed normally, you'll see some `.xml` being put into the `input` directory.
3. run this example mapreduce job:
    ```powershell
    hadoop jar "$($env:HADOOP_HOME)/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.2.2.jar" grep input output 'dfs[a-z.]+'
    ```