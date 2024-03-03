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
#include<tuple>


using namespace std;
const int backLog = 4;
const int maxDataSize = 1460;

int main()
{  
   string buffer;          // This buffer will be used to store the metadata from bootstrap server
   int check;        	   // we use this integer to confirm succesful socket function calls
   uint16_t udpPort=3002;  // we are using different ports to manage the UDP and TCP connection
   uint16_t tcpPort=3003;  // By default client knows the port number of Bootstrap server
   
   socklen_t sinSize = sizeof(struct sockaddr_in);  // all the datastructure that will be used
   int flags = 0;                                   // throughout the code is defined here to easy reference
   int dataRecvd = 0, dataSent = 0;
   struct sockaddr_in clientAddressInfo;
   char rcvDataBuf[maxDataSize];
   char sendDataBuf[maxDataSize]; 
   string sendDataStr;
   string rcvDataStr;   
   //-------------------------------------------------------------------
   // The metadata from bootstrap server will be received in a single string from bootstrap
      string server_type;                        // We will breakdown the final string
      string service_type;  					 // to these small strings
	  string ip_addr;     						 // IP Addr of the file server we want to connect
	  string port_no; 	                         // Port number of the file server we want to connect 
	  string access_token;                       // Access token that we gathered from bootstrap server
   //-------------------------------------------------------------------
   
   string serverIpAddr = "127.0.0.1";                        // this is the IP of the BOOT strap server - 
   cout<<": : : : : : : : : : B CLIENT : : : : : : : :\n\n"; // the client is supposed to know this in advance

   int udpSocketFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);  // creating a socket
   if(!udpSocketFd)
        { cout<<"Error creating UDP socket"<<endl; exit(1); } 
   else { cout<<"UDP socket created successfully . . . \n"; }

   struct sockaddr_in udpSockAddressInfo;
   udpSockAddressInfo.sin_family = AF_INET;
   udpSockAddressInfo.sin_port = htons(udpPort);
   inet_pton(AF_INET, serverIpAddr.c_str(), &(udpSockAddressInfo.sin_addr));
   memset(&(udpSockAddressInfo.sin_zero), '\0', 8);
   
   // Binding our socket
   check = bind(udpSocketFd, (struct sockaddr *)&udpSockAddressInfo, sizeof(struct sockaddr)); // binding the socket
   if(!udpSocketFd)
     { cout<<"Error binding UDP socket ! ! ! \n"<<endl;
       close(udpSocketFd);   exit(1);}
   else { cout<<"UDP Binding Successful . . .\n\n"; }    
   
   // connecting with bootstrap
   check = connect(udpSocketFd, (struct sockaddr *)&udpSockAddressInfo, sizeof(struct sockaddr));
   if (!udpSocketFd)
     { cout<<"Error with server connection "<<endl;
       close(udpSocketFd); exit(1);
     }
   else { cout<<"B CLIENT connected with BOOTSTRAP . . .\n"; }  // Confirmation for connection establishment with bootstrap
   
   // -------------------------------------------------------------------------//
   //------------  Recieving the metadata from bootstrap  ---------------------//
     check = recv(udpSocketFd, &rcvDataBuf, maxDataSize, flags);
      if(check <=0) 
         { cout<<"Meta data not recieved . . .\n"<<endl; }
      else 
      { rcvDataStr = rcvDataBuf;
        cout<<"MetaData received : -> \n"<<rcvDataStr.c_str()<<endl;  // what we recieved from server as our metadata
        cout<<"**********************************************************\n";
      } 												
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////// heavy lifting done now trying for direct file server transfer///
   
    buffer = rcvDataStr.c_str();                            // buffer received from bootstrap server
//   cout<<"printing buffer for you\n"<<buffer;             // if we want we can print and see what metadata we have revieved
     close(udpSocketFd);  // since metadata transfer is complete we dont need the connection anymore so we close it

//-------------------------------------------------------------------------------------------------------------------

  int file_server_count = 2;   // we can increase this number based on how many server we need to connect 
                               // the code is capable of handling any number of file retrival to maintain simplicity  
                               // we have set it as 2, any number it can handle
                               
// about the for loop
// This for loop is meant to do certain things as follows ->
// 1. Breakup the metadata string for one particular server and seperate different components like IP, port, token etc
// 2. Once we have the components we establish connection with file server using those attributes
// 3. after connection is established we send the ACCESS_TOKEN revieved from bootstrap server
// 4. The file server verifies our token and sends back YES/NO
// 5. Once we get yes we directly ask for respective file
// 6. after we recive the file we print a confirmation
// 7. we close the respective connection with that particular file server
// 8. we clear buffers
// 9. repeat from step 1. again
                          
   for (int i =0; i< file_server_count; i++)  // the for loop
  {
     // ...........breaking up the data bootstraps metadata.............
      int m = 38;
      server_type  = buffer.substr ((m*i)+0, 11);
      service_type = buffer.substr ((m*i)+12, 5);
	  ip_addr      = buffer.substr ((m*i)+18, 9);  // here is our new IP_address
	  port_no 	   = buffer.substr ((m*i)+28, 4);  // here is out new Port_no
	  access_token = buffer.substr ((m*i)+33, 4);
     
     //.................................................................
     // setting up connection parameters
     serverIpAddr = ip_addr;      // we got the IP address of file server to download file
     stringstream s_num(port_no); // converting string port number to integer value
     s_num >> tcpPort;            // we got the port number of server to download file
     
     // to check if we have seperated we manually display them for confirmation
     cout<<"---------------------------------\n";
     cout<<"server_type  : "<<server_type<<endl;
     cout<<"service_type : "<<service_type<<endl;
     cout<<"ip_addr      : "<<serverIpAddr<<endl;
     cout<<"port_no      : "<<tcpPort<<endl;
     cout<<"access_token : "<<access_token<<endl;   
   
//-----------------------------------------------------------------------   
   int tcpSocketFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);  // establishing a connection with file server
   if(!tcpSocketFd)
        { cout<<"Error creating TCP socket"<<endl; exit(1); } 
   else { cout<<"TCP socket created successfully . . . \n"; }

   struct sockaddr_in tcpSockAddressInfo;
   tcpSockAddressInfo.sin_family = AF_INET;
   tcpSockAddressInfo.sin_port = htons(tcpPort);
   inet_pton(AF_INET, serverIpAddr.c_str(), &(tcpSockAddressInfo.sin_addr));
   memset(&(tcpSockAddressInfo.sin_zero), '\0', 8);
   
   // binding our TCP socket
   check = bind(tcpSocketFd, (struct sockaddr *)&tcpSockAddressInfo, sizeof(struct sockaddr));  // 
   if(!tcpSocketFd)
     { cout<<"Error binding TCP socket ! ! ! \n"<<endl;
       close(tcpSocketFd);   exit(1);}
   else { cout<<"TCP Binding Successful . . .\n\n"; }
         
  // establishing connection with file server
   check = connect(tcpSocketFd, (struct sockaddr *)&tcpSockAddressInfo, sizeof(struct sockaddr));
   if (!tcpSocketFd)
     { cout<<"Error with file server connection "<<endl;
       close(tcpSocketFd); exit(1);
     }
   else { cout<<"B_CLIENT connected with FILE SERVER . . .\n"; }  // displaying confirmation for connection with file server
 //---------------------------------------------------------------------
    // sending the access_token to file server to verify out authenticity
    // if something wrong is passed the client will close the connction abruptly for saftey reasons
     sendDataStr = access_token;
     check = send(tcpSocketFd, sendDataStr.c_str(), sendDataStr.length(), flags);
     if(check == -1) cout<<"Error sending access token\n"<<endl;
     else {cout<<"Access Token sent for authentication : "<<sendDataStr<<" \n\n";}
 
  //---------------------------------------------------------------------  
 
      memset(&rcvDataBuf, 0, maxDataSize);
      check = recv(tcpSocketFd, &rcvDataBuf, maxDataSize, flags);  // recieving the clients response for our sent token
      if(check <=0 ) 
        { cout<<"Did not recieve acknowledgement ! ! !";}
      else 
       {   
		   string server_resp = rcvDataBuf;
		   string YES = "YES";
		   cout<<"File servers response for authentication : "<<server_resp<<endl; // printing what the file server responded
		   if(!(server_resp == "YES") )
		     {  cout<<"Token Rejected contact Bootstrap\n"; } // incorrect access token 
		   else
		     {  
				 cout<<"Token Accepted\n";                    // token verified getting ready to receive our file
				 //sendDataStr = "Authentication Successful\n";
				 cout<<"Recieving file from file server \n";
				 //------------ since the token i accepted we must now try for file sending
				 ofstream my_file;
                 for(int i=0;i<maxDataSize;i++)  //clearing out the buffer
                 {
			      rcvDataBuf[i] = '\0';
	             }
				 memset(rcvDataBuf, '\0', maxDataSize);
				 //-----------------------------------------------------
				 // now this is important part here we decide which type of file server is there
				 // and we decide this by comparing they filetype received from bootstrap's metadata
				 if(service_type == "texts")
				  {  my_file.open("texts.txt",std::ios::binary);}
				 else if (service_type == "image")
				  {  my_file.open("image.png",std::ios::binary);}
				 else if (service_type == "video")
				  {  my_file.open("video.mp4",std::ios::binary);}
				 else if (service_type == "_pdf_")
				  {  my_file.open("_pdf_.pdf",std::ios::binary);}
				 
				 // once we know the file type we create an empty file 
				 // on our client side with the respective extension
				 //-----------------------------------------------------
				 	
                 recv(tcpSocketFd, rcvDataBuf, maxDataSize, flags);  // receiving the file buffer from file server
                 my_file<<rcvDataBuf;                                //  writing that data into out newly created file
                 my_file.close();                                    // closing the file
                 cout<<"File transfer complete \n";
                 memset(rcvDataBuf, '\0', maxDataSize);              // clearing out the buffer for later use in the loop
				 //-----------------------------------------------------
				 close(tcpSocketFd);                                 // closing the socket for respective fileserver 
				 cout<<"Closing connection from file server\n";      // displaying confirmation for closed socket
			}
	   }
	 }  	   
   return 0;

}


