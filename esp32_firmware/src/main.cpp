// Raw continuous Float32 replay -> causal filtering -> overlapping CS packets.
#include <Arduino.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <esp_system.h>
#include <esp_arduino_version.h>
#include <math.h>
#include "cs_matrices.h"
#include "stream_config.h"

#pragma pack(push,1)
struct PacketHeader {uint8_t version,flags;uint16_t config;uint32_t frame;float scale;uint32_t crc;};
struct Timing {uint32_t raw_end,filter_us,normalize_us,project_us,quantize_us,packet_us,encoder_us,free_heap,stack_free;};
#pragma pack(pop)
static_assert(sizeof(PacketHeader)==16,"header size");
static float ring_values[256], normalized[256], filtered_trace[256], measurements[128];
static float zi[2][2];
static uint8_t packet[272];
static char command[160];
static uint32_t total_raw=0,raw_count=0,frame_count=0,filter_accum=0;
static uint16_t ring_pos=0;
static bool streaming=false,trace_mode=false;
static const StreamConfig *active=nullptr;

uint32_t crc_update(uint32_t crc,const uint8_t *p,size_t n){
  for(size_t i=0;i<n;i++){crc^=p[i];for(int bit=0;bit<8;bit++)crc=(crc>>1)^((crc&1)?0xEDB88320:0);}
  return crc;
}
// Host and device must compute CRC on precisely the same echoed bytes.
int hex_nibble(char c){
  if(c>='0'&&c<='9')return c-'0';
  if(c>='a'&&c<='f')return c-'a'+10;
  if(c>='A'&&c<='F')return c-'A'+10;
  return -1;
}
void crc_challenge(const char *hex){
  size_t n=strlen(hex);uint8_t bytes[64];
  if(!n || n%2 || n>sizeof(bytes)*2){Serial.println("ERR_CRC_INPUT");return;}
  for(size_t i=0;i<n/2;i++){
    int high=hex_nibble(hex[2*i]),low=hex_nibble(hex[2*i+1]);
    if(high<0 || low<0){Serial.println("ERR_CRC_INPUT");return;}
    bytes[i]=(uint8_t)((high<<4)|low);
  }
  uint32_t crc=~crc_update(0xFFFFFFFF,bytes,n/2);
  Serial.printf("CRC32 %s %08lX\n",hex,(unsigned long)crc);
}
float filter_one(float v){
  for(int s=0;s<2;s++){
    float w=SOS_COEFFS[s][0]*v+zi[s][0];
    zi[s][0]=SOS_COEFFS[s][1]*v-SOS_COEFFS[s][4]*w+zi[s][1];
    zi[s][1]=SOS_COEFFS[s][2]*v-SOS_COEFFS[s][5]*w;v=w;
  }
  return v;
}
void project(int M){
  const uint8_t *b=M==51?(const uint8_t*)B_PACKED_51:M==77?(const uint8_t*)B_PACKED_77:(const uint8_t*)B_PACKED_128;
  float scale=M==51?CS_INV_SQRT_M_51:M==77?CS_INV_SQRT_M_77:CS_INV_SQRT_M_128;
  for(int i=0;i<M;i++){
    float sum=0;for(int j=0;j<256;j++)sum+=((b[i*32+j/8]>>(j%8))&1)?normalized[j]:-normalized[j];
    measurements[i]=sum*scale;
  }
}
void emit_frame(){
  int M=active->M;Timing tm{};tm.raw_end=raw_count;tm.filter_us=filter_accum;filter_accum=0;
  uint64_t start=esp_timer_get_time(),t=start;
  for(int j=0;j<256;j++){
    float f=ring_values[(ring_pos+j)%256];filtered_trace[j]=f;normalized[j]=(f-active->mu)/active->sigma;
  }
  tm.normalize_us=esp_timer_get_time()-t;start=esp_timer_get_time();t=start;project(M);
  tm.project_us=esp_timer_get_time()-t;t=esp_timer_get_time();
  float a=0;for(int i=0;i<M;i++)a=fmaxf(a,fabsf(measurements[i]));
  double exact=fmax((double)a/32767.0,1e-12);float delta=(float)exact;
  if((double)delta<exact)delta=nextafterf(delta,INFINITY);if(a==0)delta=1;
  auto q=(int16_t*)(packet+16);
  for(int i=0;i<M;i++)q[i]=(int16_t)fmaxf(-32767,fminf(32767,roundf(measurements[i]/delta)));
  tm.quantize_us=esp_timer_get_time()-t;t=esp_timer_get_time();
  auto h=(PacketHeader*)packet;h->version=1;h->flags=0;h->config=active->id;h->frame=frame_count;h->scale=delta;
  uint32_t crc=crc_update(0xFFFFFFFF,packet,12);h->crc=~crc_update(crc,packet+16,2*M);
  tm.packet_us=esp_timer_get_time()-t;tm.encoder_us=esp_timer_get_time()-start;
  tm.free_heap=heap_caps_get_free_size(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT);
  // ESP-IDF uxTaskGetStackHighWaterMark returns bytes, unlike upstream FreeRTOS.
  tm.stack_free=uxTaskGetStackHighWaterMark(nullptr);
  uint16_t kind=trace_mode?2:1;uint32_t length=16+2*M+sizeof(tm)+(trace_mode?(512+M)*4:0);
  Serial.write((const uint8_t*)"CSF1",4);Serial.write((uint8_t*)&kind,2);Serial.write((uint8_t*)&length,4);
  Serial.write(packet,16+2*M);Serial.write((uint8_t*)&tm,sizeof(tm));
  if(trace_mode){Serial.write((uint8_t*)filtered_trace,1024);Serial.write((uint8_t*)normalized,1024);Serial.write((uint8_t*)measurements,4*M);}
  frame_count++;
}
void info(){
  Serial.printf("INFO_START\nCHIP=ESP32-S3\nREV=%u\nCPU_MHZ=%u\nFLASH_BYTES=%u\nPSRAM_BYTES=%u\n",ESP.getChipRevision(),ESP.getCpuFreqMHz(),ESP.getFlashChipSize(),ESP.getPsramSize());
  Serial.printf("ARDUINO_VERSION=%u.%u.%u\nESP_IDF_VERSION=%s\nFIRMWARE_VERSION=eda-raw-stream-crc-v1\n",
    ESP_ARDUINO_VERSION_MAJOR,ESP_ARDUINO_VERSION_MINOR,ESP_ARDUINO_VERSION_PATCH,esp_get_idf_version());
  Serial.printf("INTERNAL_FREE=%u\nINTERNAL_MIN_FREE=%u\nINTERNAL_LARGEST=%u\nPSRAM_FREE=%u\nSTACK_FREE_BYTES=%u\n",
    heap_caps_get_free_size(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT),heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT),
    heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT),heap_caps_get_free_size(MALLOC_CAP_SPIRAM),uxTaskGetStackHighWaterMark(nullptr));
  Serial.printf("APP_BUFFER_BYTES=%u\nMATRICES_FLASH_BYTES=%u\nPROTOCOL_SHA=%s\nREGISTRY_SHA=%s\nINFO_END\n",
    sizeof(ring_values)+sizeof(normalized)+sizeof(filtered_trace)+sizeof(measurements)+sizeof(zi)+sizeof(packet)+sizeof(command),
    sizeof(B_PACKED_51)+sizeof(B_PACKED_77)+sizeof(B_PACKED_128),PROTOCOL_SHA,REGISTRY_SHA);
}
void setup(){Serial.setRxBufferSize(4096);Serial.begin(115200);Serial.setTimeout(5000);delay(300);Serial.println("AUDITED_RAW_STREAM_READY");}
void loop(){
  if(!streaming){
    if(!Serial.available()){delay(1);return;}
    size_t n=Serial.readBytesUntil('\n',command,sizeof(command)-1);command[n]=0;
    if(strcmp(command,"INFO")==0){info();return;}
    if(strncmp(command,"CRC32 ",6)==0){crc_challenge(command+6);return;}
    unsigned id,count,trace;
    if(sscanf(command,"STREAM_BEGIN %u %u %u",&id,&count,&trace)==3){
      active=nullptr;for(const auto &cfg:CONFIGS)if(cfg.id==id)active=&cfg;
      if(!active || !count || trace>1){Serial.println("ERR_CONFIG");return;}
      memset(zi,0,sizeof(zi));memset(ring_values,0,sizeof(ring_values));ring_pos=0;raw_count=frame_count=filter_accum=0;
      total_raw=count;trace_mode=trace;streaming=true;Serial.printf("READY %u %u %s\n",id,count,REGISTRY_SHA);return;
    }
    if(strcmp(command,"STREAM_END")==0){Serial.println("OK_END");return;}
    Serial.println("ERR_COMMAND");return;
  }
  float value;
  if(Serial.readBytes((char*)&value,4)!=4){streaming=false;Serial.println("ERR_TIMEOUT");return;}
  if(!isfinite(value)){streaming=false;Serial.println("ERR_NONFINITE");return;}
  uint64_t t=esp_timer_get_time();float f=filter_one(value);filter_accum+=esp_timer_get_time()-t;
  raw_count++;
  if(raw_count>320){ring_values[ring_pos]=f;ring_pos=(ring_pos+1)%256;}
  if(raw_count>=576 && (raw_count-576)%128==0)emit_frame();
  if(raw_count==total_raw){streaming=false;Serial.printf("STREAM_DONE %u %u\n",raw_count,frame_count);}
}
