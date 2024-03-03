
  #include <bitset>
  #include <iostream>
  #include <istream>
  #include <iterator>
  #include <limits>
  #include <list>
  #include <map>
  #include <string>
  #include <typeinfo>
  #include <utility>
  #include <valarray>
  #include <vector>

using namespace std;

struct comp
{
    double real;
    double img;
};
int degree_cal(int p,int q)
{
    int x = p + q -1;

    int i=1;
    while(i<x)
    {
        i=i*2;
    }
    return i;
}

vector<double> multiply_complex_num(double a,double b,double c,double d)
{
    vector<double> mul(2);
    mul[0]= a*c - b*d;
    mul[1]= a*d + b*c;

    return mul;
}

vector<double> add_complex_num(double a,double b,double c,double d)
{
    vector<double> sum(2);
    sum[0]= a + c;
    sum[1]= b + d;
    return sum;
}

vector<double> sub_complex_num(double a,double b,double c,double d)
{
    vector<double> sum(2);
    sum[0]= a - c;
    sum[1]= b - d;
    return sum;
}


//-------FFT Start -------//

vector<comp> fft(vector<comp>& a)
{
    int n = a.size();
     vector<comp> l(1);

    // if input contains just one element
    if (n == 1)
    {
        l[0].real=a[0].real;
        l[0].img=a[0].img;
        return l;
    }
 
    vector<comp> w(n);
    for (int i = 0; i < n; i++) {
        double alpha = 2 * M_PI * i / n;
        w[i].real = cos(alpha);
        w[i].img = sin(alpha);
    }
  
    vector<comp> A0(n / 2), A1(n / 2);
    for (int i = 0; i < n / 2; i++) {
 
        // even indexed coefficients
        A0[i] = a[i * 2];
 
        // odd indexed coefficients
        A1[i] = a[i * 2 + 1];
    }
 
    // Recursive call for even indexed coefficients
    vector<comp> y0 = fft(A0);
 
    // Recursive call for odd indexed coefficients
    vector<comp> y1 = fft(A1);
 
    // for storing values of y0, y1, y2, ..., yn-1.
    vector<comp> y(n);
   
    for (int k = 0; k < n / 2; k++) {
        vector<double> mul=multiply_complex_num(y1[k].real,y1[k].img,w[k].real,w[k].img); 
        vector<double> sum=add_complex_num(mul[0],mul[1] , y0[k].real,y0[k].img);
        y[k].real = sum[0];
        y[k].img = sum[1];
      
        vector<double> sub = sub_complex_num(y0[k].real,y0[k].img,mul[0],mul[1]);
        y[k + n/2].real = sub[0];
        y[k + n/2].img =sub[1]; 
    }
    
    return y;
}


/////////////-----END FFT----------//////////////


///////////////--------Inverse FFT--------////////////
vector<comp> ifft(vector<comp>& a)
{
    
    int n = a.size();
    
    vector<comp> l(1);

    // if input contains just one element
    if (n == 1)
    {
        
        l[0].real=a[0].real;
        l[0].img=a[0].img;
        
       return l;
    }
 
    vector<comp> A0(n / 2), A1(n / 2);
    for (int i = 0; i < n / 2; i++)
     {
        A0[i] = a[i * 2];
        A1[i] = a[i * 2 + 1];
     }
 
    vector<comp> w(n);
    for (int i = 0; i < n; i++) {
        double alpha = -2 * M_PI * i / n;
        w[i].real = cos(alpha);
        w[i].img = sin(alpha);
    }

    vector<comp> y0 = ifft(A0);
    vector<comp> y1 = ifft(A1);
    vector<comp> y(n);
   
    for (int k = 0; k < n / 2; k++) {
        
        vector<double> mul=multiply_complex_num(y1[k].real,y1[k].img,w[k].real,w[k].img); 
        vector<double> sum=add_complex_num(mul[0],mul[1] , y0[k].real,y0[k].img);
    
        y[k].real = sum[0] ;
        y[k].img = sum[1] ;
      
        vector<double> sub = sub_complex_num(y0[k].real,y0[k].img,mul[0],mul[1]);
        y[k + n/2].real = sub[0] ;
        y[k + n/2].img = sub[1] ; 
      
    }
    
    return y;
}
//////////////-------- End pf Inverse fft---------//////////

/////-----naive approch--------////////
vector<comp> naiveapp(vector<comp> a,vector<comp> b)
{
   
   int asize=a.size();
   int bsize=b.size();
   int n= asize + bsize -1;
   vector<comp> ans(n);

   for(int i=0;i<n;i++)
   {
    int sum=0;
      for(int j=0;j<=i;j++)
      {
         sum= sum + a[j].real * b[i-j].real;
      }
      ans[i].real=sum;
      ans[i].img=0;
   }

   
   return ans;
}
/////-----naive approch END------////////
int main()
{
    int deg1;
    cout<<"enter the degree of first polynomial :";
    cin>>deg1;


    cout<<"enter the "<<deg1+1<<" coefficient of the first polynomial in increasing order of the degree of the monomials the belong to:"<<endl;
    vector<comp> a;
    for(int i=0;i<deg1+1;i++)
    {
        comp temp;
        cin>>temp.real;
        temp.img=0;
        a.push_back(temp);
    }

    int deg2;
    cout<<"enter the degree of second polynomial: ";
    cin>>deg2;
   
     vector<comp> a1;
     cout<<"enter the "<<deg2+1<<" coefficient of the secoond polynomial in increasing order of the degree of the monomials the belong to:"<<endl;
     for(int i=0;i<deg2+1;i++)
     {
        comp temp;
        cin>>temp.real;
        temp.img=0;
        a1.push_back(temp);
    }

     //----printing both polynomial----//
      cout<<"The first polynomial is:"<<endl;
        for (int i = deg1 ; i >= 0 ; i--)
        {
            if(i==deg1)
            {
                if(a[i].real != 1)
                cout<<a[i].real<<"X*"<<i;
                else
                cout<<"X*"<<i;  
            }
            else if(a[i].real!=0)
            {
                if(i==0)
                {
                    cout<<" "<<a[0].real;
                    break;
                }
                if(a[i].real<0)
                {
                    if(a[i].real != 1)
                    cout<<" - "<<a[i].real<<"X*"<<i;
                    else
                    cout<<" - "<<"X*"<<i;  
                }
                    else
                    {
                    if(a[i].real != 1)
                        {cout<<" + "<<a[i].real<<"X*"<<i;}
                    else
                        {cout<<" + "<<"X*"<<i;}  
                    }   
            }
        }
        cout<<endl;
        cout<<"The second polynomial is:"<<endl;
        for (int i =  deg2 ; i >= 0 ; i--)
        {
            if(i==deg2)
            {
                if(a1[i].real != 1)
                cout<<a1[i].real<<"X*"<<i;
                else
                cout<<"X*"<<i;  
            }
            else if(a1[i].real!=0)
            {
                if(i==0)
                {
                if(a[i].real != 1)
                    cout<<a1[i].real<<"X*"<<i;
                else
                    cout<<"X*"<<i;  
                    
                    break;
                }
                if(a1[i].real<0)
                {
                    if(a1[i].real != 1)
                    {
                        cout<<" - "<<a1[i].real<<"X*"<<i;
                    }
                    else
                    {cout<<" - X*"<<i;  }
                }
                else
                { 
                    if(a1[i].real != 1)
                        {cout<<" + "<<a1[i].real<<"X*"<<i;}
                    else
                        cout<<" + "<<"X*"<<i;  
                }  
            }
        }
        cout<<endl;
      //----End of printing both polynomial----//

      //-----calculate number root need for evaluating fft------// 
      int n= degree_cal(deg1+1,deg2+1);

  
            for(int i = deg1+1;i<n;i++)
            {
                comp temp;
                temp.real=0;
                temp.img=0;
                a.push_back(temp);
            }
           
            vector<comp> b = fft(a);
    
            for(int i = deg2+1;i<n;i++)
            {
                comp temp;
                temp.real=0;
                temp.img=0;
                a1.push_back(temp);
            }
            vector<comp> b1 = fft(a1);

    //---------naive approch-----------//
    vector<comp> naive= naiveapp(a,a1);

     cout<<"The product of the two polynomials obtained via naive polynomial multiplication is:"<<endl;

        for (int i = deg1 + deg2 ; i >= 0 ; i--)
        {
            if(i==deg1+deg2)
            {
                cout<<naive[i].real<<"X*"<<i;
            }
            else if(naive[i].real!=0)
            {
                if(naive[i].real<0)
                {
                    cout<<" - "<<(-1)*naive[i].real<<"X*"<<i;
                }
                else
                {
                     cout<<" + "<<naive[i].real<<"X*"<<i;  
                }  
            }
        }
    //---------End of naive approch---------//
 cout<<endl;

 vector<comp> c(n);

  for(int i=0;i<n;i++)
  {
      c[i].real= b[i].real * b1[i].real - b[i].img * b1[i].img ; 
      
      c[i].img = b[i].real * b1[i].img + b[i].img * b1[i].real ;
      
  }

  vector<comp> c1= ifft(c);
  cout<<"The product of the two polynomials obtained via polynomial multiplication using FFT is:"<<endl;
        for (int i = deg1 + deg2 ; i >= 0 ; i--)
        {
            if(i==deg1+deg2)
            {
                if(c1[i].real/n == 1)
                cout<<"X*"<<i;
                else if(c1[i].real/n == -1)  
                cout<<"X*"<<i;
                else 
                cout<<c1[i].real/n<<"X*"<<i;  
            }
            else if(c1[i].real!=0)
            {
                if(c1[i].real<0)
                {
                    if(c1[i].real/n == -1)  
                    cout<<" - X*"<<i;
                    else 
                    cout<<" - "<<(-1)*c1[i].real/n<<"X*"<<i;  
                }
                else
                {
                
                if(c1[i].real/n == 1)  
                    cout<<" + X*"<<i;
                    else 
                    cout<<" + "<<c1[i].real/n<<"X*"<<i; 
                }
            }
        }
  cout<<endl;
    return 0;
}

 
