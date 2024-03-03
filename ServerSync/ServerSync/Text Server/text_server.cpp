#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <cstring>
#include <fstream>
#include <bits/stdc++.h>

using namespace std;
const int backLog = 4;
const int maxDataSize = 1460;

int main()
{
   int check;
   string access_token = "ABCD";  // This is access token of file server
   string client_token;		      // This is what client has send 
   uint16_t udpPort = 3002;       // we will collect this and compare for authentication
   uint16_t tcpPort = 3003;       // we are using different sockets to connect to bootstrap and client 
   
   socklen_t sinSize = sizeof(struct sockaddr_in);					  //
   int flags = 0; 													  //
   struct sockaddr_in clientAddressInfo; 							  //
   char rcvDataBuf[maxDataSize]; 									  //
   char sendDataBuf[maxDataSize]; 									  //
   string sendDataStr; 												  //
   string rcvDataStr;   											  //
   
   string serverIpAddr = "127.0.0.1";      //  this is the IP Addr of bootstrap server
   cout<<": : : : : : : : : : : : TEXT SERVER : : : : : : : : : : : :\n";
   cout<<": : : : : : : : : : : : : : : : : : : : : : : : : : : : : :\n";

   int udpSocketFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);    // creating new socket for out file server
   if(!udpSocketFd)
        { cout<<"Error creating UDP socket"<<endl; exit(1); } 
   else { cout<<"UDP socket created successfully . . . \n"; }

   struct sockaddr_in udpSockAddressInfo;
   udpSockAddressInfo.sin_family = AF_INET;
   udpSockAddressInfo.sin_port = htons(udpPort);
   inet_pton(AF_INET, serverIpAddr.c_str(), &(udpSockAddressInfo.sin_addr));
   memset(&(udpSockAddressInfo.sin_zero), '\0', 8);

   check = bind(udpSocketFd, (struct sockaddr *)&udpSockAddressInfo, sizeof(struct sockaddr));  // binding our socket
   if(!udpSocketFd)
     { cout<<"Error binding UDP socket ! ! ! \n"<<endl;
       close(udpSocketFd);   exit(1);}
   else { cout<<"UDP Binding Successful . . .\n\n"; }    
   
   check = connect(udpSocketFd, (struct sockaddr *)&udpSockAddressInfo, sizeof(struct sockaddr)); // establishing connection with bootstrap
   if (!udpSocketFd)
     { cout<<"Error with server connection "<<endl;
       close(udpSocketFd); exit(1);
     }
   else { cout<<"TEXT Server connected with BOOTSTRAP . . .\n"; }  // Confirmatino that we are connected with bootstrap
   
   // ----------------------------------------------------------------//
   //------------  Send metadata to Bootstrap  -----------------------//
     sendDataStr="text_server|texts|127.0.0.1|3003|ABCD\n";  // we are using simple string with "|" as delimiter  
     check = send(udpSocketFd, sendDataStr.c_str(), sendDataStr.length(), flags);
     if(check==-1) { cout<<"Error sending metadata to Bootstrap"<<endl;}
     else          {cout<<"Metadata sent successfully to bootstrap \n";}  // confirmation that metadata is sent to bootstrap serverd
     
   //-----------------------------------------------------------------//
   //------------ Recieve Acknowledgement from server ----------------// 
     check = recv(udpSocketFd, &rcvDataBuf, maxDataSize, flags);
     if(check <=0)    { cout<<"Meta data not recieved . . .\n"<<endl;  }
     else             { rcvDataStr = rcvDataBuf;
                        cout<<"MetaData Ack : -> "<<rcvDataStr.c_str();}  // we will print the data also, to double check that what 
																		  // has been recieved by server is correct only
     cout<<"Meta data sent to Bootstrap, closing UDP port\n";
     close(udpPort);              // closing the UDP port as we need to now connect on TCP with client server for file transfer                           
////////////////////////////////////////////////////////////////////////
/////// heavy lifting is done now connecting to client//////////////////
////////////////////////////////////////////////////////////////////////

   cout<<": : : : : : : : : : TEXT SERVER READY FOR CLIENT : : : : : : : :\n\n";
   cout<<": : : : : : : : : : : : : : : : : : : : : : : :  : : : : : : : :\n\n";
   int tcpSocketFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);  // Creating TCP socket
   if(!tcpSocketFd)
        { cout<<"Error creating TCP socket"<<endl; exit(1); } 
   else { cout<<"TCP socket created successfully . . . \n"; }

   struct sockaddr_in tcpSockAddressInfo;
   tcpSockAddressInfo.sin_family = AF_INET;
   tcpSockAddressInfo.sin_port = htons(tcpPort);
   inet_pton(AF_INET, serverIpAddr.c_str(), &(tcpSockAddressInfo.sin_addr));
   memset(&(tcpSockAddressInfo.sin_zero), '\0', 8);

   check = bind(tcpSocketFd, (struct sockaddr *)&tcpSockAddressInfo, sizeof(struct sockaddr));  // binding the socket
   if(!tcpSocketFd)
     { cout<<"Error binding TCP socket ! ! ! \n"<<endl;
       close(tcpSocketFd);   exit(1);}
   else { cout<<"TCP Binding Successful . . .\n\n"; }   
   
   check = listen(tcpSocketFd, backLog);  // start listining for new connections from clients
   if(!tcpSocketFd)
     { cout<<"Error listening on TCP socket"<<endl;
       close(tcpSocketFd); exit(1); }
   else {cout<<"Listining for client to connect and ask for files. . .\n\n";} 

   int B_ClientFd = accept(tcpSocketFd, (struct sockaddr *)&clientAddressInfo, &sinSize); 
   if (!B_ClientFd)  // we have found a client to connect so we accept the connection
    {
      cout<<"Error with new file server connection :"<<endl;
      close(tcpSocketFd); exit(1);
    }
    cout<<"connected with B_CLIENT * \n"<<endl;   // confirmation message for connection with client
///////// Connenction with client done now passing access token ////////
//....................................................................//
     
    memset(&rcvDataBuf, 0, maxDataSize);  // the first message the we get from client is access_token
    check = recv(B_ClientFd, &rcvDataBuf, maxDataSize, flags);
    if(check <=0 ) 
      { cout<<"Did not recieve cliets Token ! ! !";}
    else 
      {   
		client_token = rcvDataBuf;                         // the client's token is stored in client_token
		cout<<"Server Access Token : "<<access_token<<endl;// servers token is stored in access_token
		cout<<"Client Access Token : "<<rcvDataBuf<<endl;
		//sendDataStr = access_token;
//....................................................................//	
		       		   
    if(!strcmp(access_token.c_str(), client_token.c_str()) )   // comparing both the values
	  { 
		cout<<"Access Token matched\n"; 
		sendDataStr = "YES";                   }   // values matched authentication successful
	else
      {  
		cout<<"Access Token did not match\n";      // incase incorrect token is send we notify the client by sending "NO"
		sendDataStr = "NO";                    }
		     
    check = send(B_ClientFd, sendDataStr.c_str(), sendDataStr.length(), flags);
           
    if(check == -1) 
         { cout<<"Error sending acknowledgement\n"<<endl;}
    else 
         { cout<<"Acknowledgement sent to client : "<<sendDataStr<<" \n\n"; }
           
           
//------------------- Sending the file to server  ----------------------
           
    cout<<"Sending  file to server \n"; 
    ifstream my_file;
    my_file.open("text.txt",std::ios::binary);  
    my_file.seekg(0,ios::end);
    int length = my_file.tellg();
    my_file.seekg(0, ios::beg);
    my_file.read(sendDataBuf,length);
    send(B_ClientFd, sendDataBuf, length, flags);
    my_file.close();
    cout<<"File sending completed . . .  \n";
  }
 
 return 0;    
}


