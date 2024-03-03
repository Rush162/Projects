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

main()
{
   string buffer;              // buffer to store the metadata about file server
   int check;                  // variable to check function calling status
   uint16_t udpPort = 3002;    // We are using different ports to distiguish -
   uint16_t tcpPort = 3003;    // between connection types

   socklen_t sinSize = sizeof(struct sockaddr_in);  // all the data structures
   int flags = 0;                                   // that will be used is defined
   int dataRecvd = 0, dataSent = 0;                 // here 
   struct sockaddr_in clientAddressInfo;
   char rcvDataBuf[maxDataSize];
   char sendDataBuf[maxDataSize];
   string sendDataStr;
   string recvDataStr;

   string serverIpAddr = "127.0.0.1";     // this is the IP fixed for bootstrap server
   cout<<": : : : : : : : : : BOOTSTRAP SERVER : : : : : : : :\n";
   cout<<": : : : : : : : : : : : : : : : : : :: : : : : : : :\n";
                                                        //////////
   int udpSocketFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP); //  Basic socket creation
   if(!udpSocketFd)                                             //
        { cout<<"Error creating UDP socket"<<endl; exit(1); }   //
   else { cout<<"UDP socket created successfully . . . \n"; }   //
                                                        //////////
   struct sockaddr_in udpSockAddressInfo;
   udpSockAddressInfo.sin_family = AF_INET;
   udpSockAddressInfo.sin_port = htons(udpPort);
   inet_pton(AF_INET, serverIpAddr.c_str(), &(udpSockAddressInfo.sin_addr));                    //
   memset(&(udpSockAddressInfo.sin_zero), '\0', 8);                                             //
                                                                                                //
   check = bind(udpSocketFd, (struct sockaddr *)&udpSockAddressInfo, sizeof(struct sockaddr));  // Binding the socket
   if(!udpSocketFd)
     { cout<<"Error binding UDP socket ! ! ! \n"<<endl;
       close(udpSocketFd);   exit(1);}
   else { cout<<"UDP Binding Successful . . .\n\n"; }

   check = listen(udpSocketFd, backLog);          // start listining for new connections from fileservers
    if(!udpSocketFd)
      { cout<<"Error listening on UDP socket"<<endl;
        close(udpSocketFd); exit(1); }
    else {cout<<"Listining for file server to register. . .\n\n";}

// . . . . . basic requirements done now resistering data servers ......//

   int file_server_count = 2;   // To reduce testing time i am using only 2 servers
                                // the code is capable of running any number of servers
  while(file_server_count--)    // This while loop will accept the servers in sequencial order
   {
     memset(&clientAddressInfo, 0, sizeof(struct sockaddr_in));
     memset(&rcvDataBuf, 0, maxDataSize);

     int newClientFd = accept(udpSocketFd, (struct sockaddr *)&clientAddressInfo, &sinSize);  // accepting new data servers
     if (!newClientFd)
     {
        cout<<"Error with new file server connection :"<<endl;
        close(udpSocketFd);  exit(1);
     }
    cout<<"File Server : "<<file_server_count<<" docked sucessfully "<<endl;  // displaying the confirmation for docking
    // -----------------------------------------------------------------------//
    //------------  Recieve Function for metadata-----------------------------//
      memset(&rcvDataBuf, 0, maxDataSize);
      check = recv(newClientFd, &rcvDataBuf, maxDataSize, flags);
      if(check <=0 )  { cout<<"Did not recieve server's metadata ! ! !";}
      else            { cout<<"Metadata : "<<rcvDataBuf;                }
    //..................................................................
       buffer = buffer + rcvDataBuf;  // Concatinating all the buffers from all file servers
    //------------------------------------------------------------------------// so that we can send them in one go to clients
    //------------  Acknowledge to File Server about reciving metadata--------//
     sendDataStr=rcvDataBuf;
     check = send(newClientFd, sendDataStr.c_str(), sendDataStr.length(), flags);
     if(check == -1) cout<<"Error sending Acknowledgement"<<endl;
     else {cout<<"Ack sent to data server \n\n";}
    //----------------------------------------------------------------//
  }  //the while loop ends here, we now have all the metadata

   cout<<"All fileserver's metadata acquired\n  . . .\n\n\n"<<endl;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//............we have the buffer ready to send to client.............. /////////
//cout<<buffer;          // if we want we can display whatever we have gathered/
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//......creating new socket to connect to a client ...........................//

     memset(&clientAddressInfo, 0, sizeof(struct sockaddr_in));
     memset(&rcvDataBuf, 0, maxDataSize);
     int new_B_ClientFd = accept(udpSocketFd, (struct sockaddr *)&clientAddressInfo, &sinSize);  //accepting incoming connecitons
     if (!new_B_ClientFd)
     {
        cout<<"Error with new B_CLIENT server connection :"<<endl;
        close(udpSocketFd);  exit(1);
     }
    cout<<"B_CLIENT Connect, ready to data transfer . . .\n "<<endl;  // now the client is connected and ready to recieve
                                                                      // the metadata from bootstrap server

     sendDataStr=buffer;   // copying the buffer to sendDataStr to maintain uniformity across code
     check = send(new_B_ClientFd, sendDataStr.c_str(), sendDataStr.length(), flags);
     if(check == -1){  cout<<"Error sending buffer to B_CLIENT\n"<<endl; }
     else           {  cout<<"Buffer sent to B_CLIENT . . . . \n\n";     }

     close(udpSocketFd);  //  closing the socket as our job is done

return 0;
}
