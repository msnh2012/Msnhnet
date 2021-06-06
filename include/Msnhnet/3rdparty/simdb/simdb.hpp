
/*
  Copyright 2017 Simeon Bassett

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

/*
 SimDB

 What it does:
 |  SimDB is a key value store that uses arbitrary byte data (of arbitrary length) as both the key and the value. 
 |  It additionally uses shared memory, which allows processes to communicate with each other quickly.  
 |  It is lock free and scales well with multiple threads writing, reading, and deleting concurrently.  

 How it works:
 |-simdb:
   | This contains the user facing interface. It contains the ConcurrentHash, ConcurentStore, and SharedMem classes as members.
   | These data structures are made to be an interface over the contiguous memory given to them using a single address. 
   | They do not allocate any heap memory themselves, but do have a few class members that will be on the stack. At the time of this writing it is 176 bytes on the stack.
   |-SharedMem:
   |  |  Interface to OS specific shared memory functions.  Also handles an initial alignment.
   |-ConcurrentHash:
   |  |  Hash map that uses atomic operations on an array of VerIdx structs. 
   |  |  It uses 64 bit atomic operations to compare-exchange one VerIdx at a time (VerIdx is two unsigned 32 bit integers, a version and an index). 
   |  |  This makes sure that reading, writing and deleting is lock free. 
   |  |  Writing is lock free since a VerIdx is already fully created and written to before putting it in the VerIdx array (m_vis) and the put operation here is a single 64 bit compare and swap.    
   |  |  Deletion is lock free since the index in VerIdx is only freed from the CncrLst after setting the VerIdx here to DELETED. Actually deletion means 1. setting the VerIdx to DELETED 2. decrementing the readers of the blocklist that idx points to 3. If the readers variable of that blocklist is decremented below its initial value then the thread that took it below its initial value is the one to free it. 
   |  |  Get is lock free since it can read an index from a VerIdx, increment readers, compare its key to the key in the list of blocks, read the value in the blocks to the output buffer and finally decrement the readers variable. Just like deletion, if a thread decrements readers below its initial value, it needs to free the block list.  This means the last one out cleans up.
   |-ConcurrentStore:
   |  |  Keeps track of block lists.
   |  |  This primarily uses an array of BlkLst structs (which are 24 bytes each). 
   |  |  The BlkLst lava_vec is used to make linked lists of block indices. 
   |  |  The idea of a block list ends up being a starting index (from the VerIdx struct in the concurrent hash). The BlkLst struct at the starting index contains an index of the next BlkLst struct and so on until reaching a BlkLst that has an index of LIST_END. This means that one array contains multiple linked lists (using indices and not pointers of course).
   |  |  This exposes an alloc() function and a free() function. 
   |  |  alloc() gets the index of the next block from CncrLst (concurrent list).
   |  |  The BlkLst struct keeps the total length and the key length / value offset since it does not have to be atomic and is only initialized and used when one thread allocates and only destroyed when one thread frees, just like the actual data blocks.
   |-ConcurrentList:
   |  |  The concurrent list is an array integers.
   |  |  The number of elements (like all the arrays) is the number of blocks.
   |  |  There is one integer per block with the integer at a given index representing the next slot in the list.
   |  |  The end of the list will have value LIST_END.  On initialization the array's values would be |1|2|3|4| ... LIST_END, which makes a list from the start to the end. This means s_lv[0] would return 1.

 Terms:
 |-Block List: 
 |  A sequence of block indices.  The entry in ConcurrentHash gives the position in the block list array where the list starts.  
 |  The value at each index in the array contains the index of the next block.  
 |  The list end is know when a special value of LIST_END is found as the value in the array.
 |-Block List Version:
 |  This is a version number given to each block list on allocation (not each block). 
 |  It is used to link a ConcurrentHash value to the block list. 
 |  If the versions are the same, it is known that the block list at the index read from ConcurrentHash has not changed.
 |  This change could happen if:
 |  |  1. Thread ONE reads the entry in ConcurrentHash but has not accessed the block list index in the entry yet. Pretend that thread one stalls and nothing more happens until further down.
 |  |  2. Thread TWO has already allocated a block list and swaps its new entry for the old entry which is still carried by thread one. 
 |  |  3. Thread TWO now must free the block list given by the old entry, which it does, because no thread is reading it since thread one is still stalled.
 |  |  4. Thread TWO allocates another block list, which ends up using the blocks it just deallocated.
 |  |  5. Thread ONE wakes up and reads from the block index it found in the ConcurrentHash entry, which is no longer the same and may not even be the head of the list.
 |  |  If the index is used purely for matching the binary key, this wouldn't be a problem. 
 |  |  When the index is used to find a binary value however, this is a problem, since the length of a different value could be the same, and there would be no data to be able to tell that they are different.

 How it achieves lock free concurrency:
 |  ConcurrentHash is treated as the authority of what is stored in the database. 
 |  It has an array of VerIdx structs that can also be treated as 64 bit integers. Each is dealt with atomically.
 |  Its individual bits are used as a bitfied struct containing an index into ConcurrentStore's block list as well as the version number of that list.
 |  The core is m_vis, which is an array of VerIdx structs. The memory ordering is swapped on every other index in preparation for robin hood hashing techniques, so the actual memory layout (separated into 128 bit chunks) is |Index Version Version Index|Index Version Version Index| 
 |-Finding a matching index: 
 |  |  1. Use the hash of the key bytes to jump to an index.
 |  |  2. Load the integer atomically from that index and treat it as a VerIdx struct.
 |  |  3. Use the index from that struct to read the bytes from the list of blocks in BlkLst. 
 |  |  4. Increment the readers variable atomically, so that it won't be deleted before this thread is done with it.
 |  |  5. If there is a match, keep reading the list of blocks to fill the output buffer with the value section of the block list.
 |  |  6. After, decrement the readers variable atomically.  If readers goes below its initial value, this thread will be the one to free the block list.

 Other notables:
 | All of the main classes have a static sizeBytes() function that takes in the same arguments as a constructor and return the number of bytes that it will need in the shared memory
 | Classes have member variables that are used as interfaces to the shared memory denoted with s_ (s for shared)
 | Normal member variables that are just data on the stack are denoted with m_ (m for member)

 _________________
 | Memory Layout | 
 -----------------
   ______________________________________________________________________________________________________________________
   |Flags|BlockSize|BlockCount|ConcurrentHash|ConcurrentStore|ConcurentList|...BlockCount*BlockSize bytes for blocks....|
       _____________________________/               \_______       \______________________________________________________
 ______|____________________________________   ____________|_________________________________________________    ________|___________________________________________
 |size(bytes)|...array of VerIdx structs...|   |Block List Version|size(bytes)|...array of BlkLst structs...|    |size(bytes)|...array of unsigned 32 bit ints (u32)|


 First 24 bytes (in 8 byte / unsigned 64 bit chunks): 
     ____________________________ 
     |Flags|BlockSize|BlockCount|
  
  Flags:      Right now holds count of the number of processes that have the db open.  When the count goes to 0, the last process will delete the shared memory file.
  BlockSize:  The size in bytes of a block.  A good default would be to set this to the common page size of 4096 bytes.  
  BlockCount: The number of blocks.  This hash table array, block list array and concurrent list array will all be the same length.  This multiplied by the BlockSize will give the total amount of bytes available for key and value data. More blocks will also mean the hash table will have less collisions as well as less contention between threads.

*/

// -todo: make a list cut itself off at the end by inserting LIST_END as the last value 
// -todo: look into readers and matching - should two threads with the same key ever be able to double insert into the db? - MATCH_REMOVED was not re-looping on the current index
// -todo: make MATCH_REMOVED restart the current index
// -todo: make runIfMatch return a pair that includes the return value of the function it runs
// -todo: make sure version setting on free sets the version to 0 on the whole list
// -todo: make sure incReaders and decReaders are using explicit sequential consistency - already done
// -todo: make sure that if there is a version mismatch when comparing a block list, the block list version is still used when trying to swap the version+idx - would only the index actually be needed since a block list with incremented readers won't give up its index, thus it should be unique?
// -todo: take version argument out of incReaders and decReaders
// -todo: make a temporary thread_local variable for each thread to count how many allocations it has made and how many allocations it has freed - worked very well to narrow down the problem
// -todo: make sure that the VerIdx being returned from putHashed is actually what was atomically swapped out
// -todo: try putting LIST_END at the end of the the concurrent lists - not needed for now
// -todo: debug why 2 threads inserting the same key seems to need all blocks instead of just 3 * 2 * 2 (three blocks per key * two threads * two block lists per thread) - delete flag in block lists was not always set
// -todo: assert that the block list is never already deleted when being deleted from putHashed - that wasn't the problem
// -todo: check what happens when the same key but different versions are inserted - do two different versions end up in the DB? does one version end up undeletable ?  - this was fixed by only comparing the key without the version
// -todo: check path of thread that deletes a key, make sure it replaces the index in the hash map - how do two conflicting indices in the hash map resolve? the thread that replaces needs to delete the old allocation using the version - is the version / deleted flag being changed atomically in the block list 
// -todo: change the Match enum to be an bit bitfield with flags - not needed for now
// -todo: make simdb len() and get() ignore version numbers for match and only match keys

// todo: make sure get() only increments and decrements the first/key block in the block list
// todo: make simdb give a proper error if running out of space
// todo: make simdb expand when eighther out of space or initialized with a larger amount of space
// todo: make a get function that takes a key version struct
// todo: make a get function that returns a tbl if tbl.hpp is included

#ifdef _MSC_VER
  #pragma once
  #pragma warning(push, 0)
#endif

#ifndef __SIMDB_HEADER_GUARD__
#define __SIMDB_HEADER_GUARD__

// turn asserts on an off - not sure of the best way to handle this with gcc and clang yet
#ifdef _MSC_VER
  #if !defined(_DEBUG)
    #define NDEBUG
  #endif
#endif

#if !defined(SECTION)
  #define       SECTION(_msvc_only_collapses_macros_with_arguments, ...)
#endif

// platform specific includes - mostly for shared memory mapping and auxillary functions like open, close and the windows equivilents
#if defined(_WIN32)      // windows  
  #include <locale>
  #include <codecvt>

  #include <tchar.h>

  //#ifdef UNICODE
  //  #undef UNICODE
  //#endif
  #define NOMINMAX
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <strsafe.h>

  #ifdef MIN
    #undef MIN
  #endif
  #ifdef MAX
    #undef MAX
  #endif

  #ifdef _MSC_VER
    #if !defined(_CRT_SECURE_NO_WARNINGS)
      #define _CRT_SECURE_NO_WARNINGS
    #endif

    #if !defined(_SCL_SECURE_NO_WARNINGS)
      #define _SCL_SECURE_NO_WARNINGS
    #endif
  #endif
#elif defined(__APPLE__) || defined(__MACH__) || defined(__unix__) || defined(__FreeBSD__) || defined(__linux__)  // osx, linux and freebsd
  // for mmap and munmap
  // PROT_READ and PROT_WRITE  to allow reading and writing but not executing of the mapped memory pages
  // MAP_ANONYMOUS | MAP_SHARED for the anonymous shared memory we want
  // mmap is system call 2 on osx, freebsd, and linux
  // the apple docs for mmap say "BSD System Calls" so I guess they haven't changed them around
  #include <sys/mman.h>
  #include <sys/fcntl.h>
  #include <sys/errno.h>
  #include <sys/unistd.h>
  #include <sys/file.h>         // for flock (file lock)
  #include <sys/stat.h>
  #include <sys/param.h>
  #include <unistd.h>
  #include <dirent.h>
  #include <errno.h>
#endif

#include <cstdint>
#include <cstring>
#include <atomic>
#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <cassert>

// platform specific type definitions
#ifdef _WIN32                         // these have to be outside the anonymous namespace
  typedef void        *HANDLE;
  typedef HANDLE     *PHANDLE;
  typedef wchar_t       WCHAR;        // wc,   16-bit UNICODE character
  typedef UCHAR       BOOLEAN;        // winnt
  typedef unsigned long ULONG;
#endif

//#ifndef NDEBUG
  thread_local int __simdb_allocs   = 0;
  thread_local int __simdb_deallocs = 0;
//#endif

namespace {
  enum Match { MATCH_FALSE=0, MATCH_TRUE=1, MATCH_REMOVED = -1, MATCH_TRUE_WRONG_VERSION = -2  };

  template<class T>
  class lava_noop
  {
    void operator()(){}
  };

  inline uint64_t fnv_64a_buf(void const *const buf, uint64_t len)                              // sbassett - I know basically nothing about hash functions and there is likely a better one out there 
  {
    uint64_t hval = 0xCBF29CE484222325;
    uint8_t*   bp = (uint8_t*)buf;	             // start of buffer 
    uint8_t*   be = bp + len;		                 // beyond end of buffer 
    while(bp < be){                              // FNV-1a hash each octet of the buffer
      hval ^= (uint64_t)*bp++;                   // xor the bottom with the current octet */
      hval += (hval << 1) + (hval << 4) + (hval << 5) +
              (hval << 7) + (hval << 8) + (hval << 40);
    }
    return hval;
  }
  
  inline void prefetch1(char const* const p)
  {
    #ifdef _MSC_VER                              // if msvc or intel compilers
      _mm_prefetch(p, _MM_HINT_T1);
    #elif defined(__GNUC__) || defined(__clang__)
      __builtin_prefetch(p);
    #else

    #endif
  }

  #ifdef _WIN32
    typedef struct _UNICODE_STRING {
      USHORT Length;
      USHORT MaximumLength;
      #ifdef MIDL_PASS
          [size_is(MaximumLength / 2), length_is((Length) / 2) ] USHORT * Buffer;
      #else // MIDL_PASS
          _Field_size_bytes_part_(MaximumLength, Length) PWCH   Buffer;
      #endif // MIDL_PASS
      } UNICODE_STRING;
    typedef UNICODE_STRING *PUNICODE_STRING;

    typedef struct _OBJECT_ATTRIBUTES {
        ULONG Length;
        HANDLE RootDirectory;
        PUNICODE_STRING ObjectName;
        ULONG Attributes;
        PVOID SecurityDescriptor;        // Points to type SECURITY_DESCRIPTOR
        PVOID SecurityQualityOfService;  // Points to type SECURITY_QUALITY_OF_SERVICE
    } OBJECT_ATTRIBUTES;
    typedef OBJECT_ATTRIBUTES *POBJECT_ATTRIBUTES;

    typedef long LONG;
    typedef LONG NTSTATUS;

    // the following is api poison, but seems to be the only way to list the global anonymous memory maps in windows  
    #define DIRECTORY_QUERY              0x0001  
    #define STATUS_SUCCESS               ((NTSTATUS)0x00000000L)    // ntsubauth
    #define OBJ_CASE_INSENSITIVE         0x00000040L
    #define STATUS_NO_MORE_FILES         ((NTSTATUS)0x80000006L)
    #define STATUS_NO_MORE_ENTRIES       ((NTSTATUS)0x8000001AL)

    typedef struct _IO_STATUS_BLOCK {
		  union {
			  NTSTATUS Status;
			  PVOID    Pointer;
		  };
		  ULONG_PTR Information;
	  } IO_STATUS_BLOCK, *PIO_STATUS_BLOCK;
  
    using NTOPENDIRECTORYOBJECT = NTSTATUS (WINAPI*)(
	    _Out_  PHANDLE DirectoryHandle,
	    _In_   ACCESS_MASK DesiredAccess,
	    _In_   POBJECT_ATTRIBUTES ObjectAttributes
	  );
    using NTOPENFILE = NTSTATUS (WINAPI*)(
      _Out_ PHANDLE               FileHandle,
      _In_  ACCESS_MASK        DesiredAccess,
      _In_  POBJECT_ATTRIBUTES ObjectAttributes,
      _Out_ PIO_STATUS_BLOCK   IoStatusBlock,
      _In_  ULONG                ShareAccess,
      _In_  ULONG                OpenOptions
    );
    using NTQUERYDIRECTORYOBJECT = NTSTATUS(WINAPI*)(
	    _In_       HANDLE DirectoryHandle,
	    _Out_opt_  PVOID Buffer,
	    _In_       ULONG Length,
	    _In_       BOOLEAN ReturnSingleEntry,
	    _In_       BOOLEAN RestartScan,
	    _Inout_    PULONG Context,
	    _Out_opt_  PULONG ReturnLength
	  );
    using RTLINITUNICODESTRING = VOID(*)(
      _Out_    PUNICODE_STRING DestinationString,
      _In_opt_ PCWSTR          SourceString
    );

    struct OBJECT_DIRECTORY_INFORMATION { UNICODE_STRING name; UNICODE_STRING type; };

    //auto      GetLastErrorStdStr() -> std::string
    //{
    //  DWORD error = GetLastError();
    //  if (error)
    //  {
    //    LPVOID lpMsgBuf;
    //    DWORD bufLen = FormatMessage(
    //        FORMAT_MESSAGE_ALLOCATE_BUFFER | 
    //        FORMAT_MESSAGE_FROM_SYSTEM |
    //        FORMAT_MESSAGE_IGNORE_INSERTS,
    //        NULL,
    //        error,
    //        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
    //        (LPTSTR) &lpMsgBuf,
    //        0, NULL );
    //    if (bufLen)
    //    {
    //      LPCSTR lpMsgStr = (LPCSTR)lpMsgBuf;
    //      std::string result(lpMsgStr, lpMsgStr+bufLen);
    //  
    //      LocalFree(lpMsgBuf);
    //
    //      return result;
    //    }
    //  }
    //  return std::string();
    //}
    PVOID  GetLibraryProcAddress(PSTR LibraryName, PSTR ProcName)
    {
      return GetProcAddress(GetModuleHandleA(LibraryName), ProcName);
    }
    int               win_printf(const char * format, ...)
    {
      char szBuff[1024];
      int retValue;
      DWORD cbWritten;
      va_list argptr;
          
      va_start( argptr, format );
      retValue = wvsprintfA( szBuff, format, argptr );
      va_end( argptr );

      WriteFile(  GetStdHandle(STD_OUTPUT_HANDLE), szBuff, retValue,
                  &cbWritten, 0 );

      return retValue;
    }
  #endif // end  #ifdef _WIN32
}

#ifdef _WIN32
  #pragma warning(pop)
#endif

enum class simdb_error { 
  NO_ERRORS=2, 
  DIR_NOT_FOUND, 
  DIR_ENTRY_ERROR, 
  COULD_NOT_OPEN_MAP_FILE, 
  COULD_NOT_MEMORY_MAP_FILE,
  SHARED_MEMORY_ERROR,
  FTRUNCATE_FAILURE,
  FLOCK_FAILURE,
  PATH_TOO_LONG
};

template<class T> 
class    lava_vec
{
public:
  using u64   =   uint64_t;

private:
  void* p;

  void  set_sizeBytes(u64 sb){ ((u64*)p)[-1] = sb; }                                // an offset of -2 should be the first 8 bytes, which store the size in bytes of the whole memory span of this lava_vec

public:
  static u64 sizeBytes(u64 count)                                                   // sizeBytes is meant to take the same arguments as a constructor and return the total number of bytes to hold the entire stucture given those arguments 
  {
    return sizeof(u64) + count*sizeof(T);
  }

  lava_vec(){}
  lava_vec(void*  addr, u64 count, bool owner=true) :
    p( ((u64*)addr) + 1 )
  {
    if(owner){
      set_sizeBytes( lava_vec::sizeBytes(count) );
    }
  }
  lava_vec(void*  addr) : p( ((u64*)addr) + 2 ) {}
  lava_vec(lava_vec const&)       = delete;
  void operator=(lava_vec const&) = delete;

  lava_vec(lava_vec&& rval){ p=rval.p; rval.p=nullptr; }
  ~lava_vec(){}

  T& operator[](u64 i){ return data()[i]; }

  T*        data(){ return (T*)p; }
  u64  sizeBytes() const { return ((u64*)p)[0]; }                                    // first 8 bytes should be the total size of the buffer in bytes
  auto      addr() const -> void*
  {
    return p;
  }
};
class     CncrLst
{
// Internally this is an array of indices that makes a linked list
// Externally indices can be gotten atomically and given back atomically
// | This is used to get free indices one at a time, and give back in-use indices one at a time
// Uses the first 8 bytes that would normally store sizeBytes as the 64 bits of memory for the Head structure
// Aligns the head on a 64 bytes boundary with the rest of the memory on a separate 64 byte boudary. This puts them on separate cache lines which should eliminate false sharing between cores when atomicallyaccessing the Head union (which will happen quite a bit) 
public:
  using     u32  =  uint32_t;
  using     u64  =  uint64_t;
  using    au32  =  volatile std::atomic<u32>;
  using    au64  =  volatile std::atomic<u64>;
  using ListVec  =  lava_vec<u32>;

  union Head
  {
    struct { u32 ver; u32 idx; };                           // ver is version, idx is index
    u64 asInt;
  };
  
  static const u32        LIST_END = 0xFFFFFFFF;
  static const u32 NXT_VER_SPECIAL = 0xFFFFFFFF;

//private:
  ListVec         s_lv;
  volatile au64*   s_h;

public:
  static u64   sizeBytes(u32 size) { return ListVec::sizeBytes(size) + 128; }         // an extra 128 bytes so that Head can be placed (why 128 bytes? so that the head can be aligned on its own cache line to avoid false sharing, since it is a potential bottleneck)
  static u32  incVersion(u32    v) { return v==NXT_VER_SPECIAL?  1  :  v+1; }

  CncrLst(){}
  CncrLst(void* addr, u32 size, bool owner=true)             // this constructor is for when the memory is owned an needs to be initialized
  {                                                          // separate out initialization and let it be done explicitly in the simdb constructor?    
    u64   addrRem  =  (u64)addr % 64;
    u64 alignAddr  =  (u64)addr + (64-addrRem);
    assert( alignAddr % 64 == 0 );
    s_h = (au64*)alignAddr;

    u32* listAddr = (u32*)((u64)alignAddr+64);
    new (&s_lv) ListVec(listAddr, size, owner);

    if(owner){
      for(u32 i=0; i<(size-1); ++i) s_lv[i] = i+1;
      s_lv[size-1] = LIST_END;

      ((Head*)s_h)->idx = 0;
      ((Head*)s_h)->ver = 0;
    }
  }

  bool headCmpEx( u64*expected, au64 desired)
  {
    using namespace std;

    //return atomic_compare_exchange_strong_explicit(
    //  s_h, (volatile au64*)&expected, desired,
    //  memory_order_seq_cst, memory_order_seq_cst
    //  );

    //return atomic_compare_exchange_strong(
    //  s_h, (volatile au64*)&expected, desired
    //);

      // Msnh Modified: 2021/05/26
    return atomic_compare_exchange_strong_explicit<u64>(
      s_h, expected, desired,
      memory_order_seq_cst, memory_order_seq_cst
    );

  }
  u32        nxt()                                                             // moves forward in the list and return the previous index
  {
    Head  curHead, nxtHead;
    curHead.asInt  =  s_h->load();
    do{
      if(curHead.idx==LIST_END){
        return LIST_END;
      }

      nxtHead.idx  =  s_lv[curHead.idx];
      nxtHead.ver  =  curHead.ver==NXT_VER_SPECIAL? 1  :  curHead.ver+1;
    }while( !headCmpEx( &curHead.asInt, (au64)(nxtHead.asInt) )); // Msnh Modified: 2021/05/26
    //}while( !headCmpEx(curHead.asInt, nxtHead.asInt) );
    //}while( !s_h->compare_exchange_strong(curHead.asInt, nxtHead.asInt) );

    return curHead.idx;
  }
  u32        nxt(u32 prev)                                                             // moves forward in the list and return the previous index
  {
    using namespace std;
    
    Head  curHead, nxtHead, prevHead;
    curHead.asInt  =  s_h->load();
    do{
      if(curHead.idx==LIST_END){
        return LIST_END;
      }

      prevHead     =  curHead;
      nxtHead.idx  =  s_lv[curHead.idx];
      nxtHead.ver  =  curHead.ver==NXT_VER_SPECIAL? 1  :  curHead.ver+1;
    }while( !headCmpEx( &curHead.asInt, (au64)nxtHead.asInt) );// Msnh Modified: 2021/05/26
    //}while( !headCmpEx(curHead.asInt, nxtHead.asInt) );
    //}while( !s_h->compare_exchange_strong(curHead.asInt, nxtHead.asInt) );

    //s_lv[prev] = curHead.idx;
    atomic_store( (au32*)&s_lv[prev], curHead.idx);

    return curHead.idx;
  }
  u32        free(u32 idx)                                                    // not thread safe when reading from the list, but it doesn't matter because you shouldn't be reading while freeing anyway, since the CncrHsh will already have the index taken out and the free will only be triggered after the last reader has read from it 
  {
    Head curHead, nxtHead; u32 retIdx;
    curHead.asInt = s_h->load();
    do{
      retIdx = s_lv[idx] = curHead.idx;
      nxtHead.idx  =  idx;
      nxtHead.ver  =  curHead.ver + 1;
    }while( !headCmpEx( &curHead.asInt, (au64)nxtHead.asInt) );// Msnh Modified: 2021/05/26
    //}while( !headCmpEx(curHead.asInt, nxtHead.asInt) );
    //}while( !s_h->compare_exchange_strong(curHead.asInt, nxtHead.asInt) );

    return retIdx;
  }
  u32        free(u32 st, u32 en)                                            // not thread safe when reading from the list, but it doesn't matter because you shouldn't be reading while freeing anyway, since the CncrHsh will already have the index taken out and the free will only be triggered after the last reader has read from it 
  {
    using namespace std;
  
    Head curHead, nxtHead; u32 retIdx;
    curHead.asInt = s_h->load();
    do{
      //retIdx = s_lv[en] = curHead.idx;
      retIdx = curHead.idx;
      atomic_store_explicit( (au32*)&(s_lv[en]), curHead.idx, memory_order_seq_cst);
      //atomic_store( (au32*)&(s_lv[en]), curHead.idx);
      nxtHead.idx  =  st;
      nxtHead.ver  =  curHead.ver + 1;
    }while( !headCmpEx( &curHead.asInt, (au64)nxtHead.asInt) );// Msnh Modified: 2021/05/26
    //}while( !headCmpEx(curHead.asInt, nxtHead.asInt) );
    //}while( !s_h->compare_exchange_strong(curHead.asInt, nxtHead.asInt) );

    return retIdx;
  }
  u32       alloc(u32 count)
  {
    u32   st = nxt();
    u32  cur = st;
    if(st == LIST_END) return LIST_END;
    else --count;

    while( count > 0 ){
      u32 nxtIdx = nxt(cur);
      if(nxtIdx == LIST_END){
        free(st,cur);
        return LIST_END;
      }
      cur = nxtIdx;
      --count;
    }

    //s_lv[cur] = LIST_END;
    return st;
  }
  auto      count() const -> u32 { return ((Head*)s_h)->ver; }
  auto        idx() const -> u32
  {
    Head h; 
    h.asInt = s_h->load();
    return h.idx;
  }
  auto       list() -> ListVec const* { return &s_lv; }                      // not thread safe
  u32      lnkCnt()                                                          // not thread safe
  {
    u32    cnt = 0;
    u32 curIdx = idx();
    while( curIdx != LIST_END ){
      curIdx = s_lv[curIdx];
      ++cnt;
    }
    return cnt;
  }
  auto       head() -> Head* { return (Head*)s_h; }
};
class     CncrStr                                                          // CncrStr is Concurrent Store 
{
public:
  using      u8  =  uint8_t;
  using     u32  =  uint32_t;
  using     i32  =   int32_t;
  using     u64  =  uint64_t;
  using     i64  =   int64_t;
  using    au32  =  std::atomic<u32>;
  using    au64  =  std::atomic<u64>;

  union   VerIdx
  {
    struct { u32 idx; u32 version; }; 
    u64 asInt;

    VerIdx(){}
    VerIdx(u32 _idx, u32 _version) : idx(_idx), version(_version) {}
  };
  union   KeyReaders
  {
    struct{ u32 isKey : 1; u32 isDeleted : 1; i32 readers : 30; };
    u32 asInt;
  };
  struct  BlkLst                                                   // 24 bytes total
  {    
    union{
      KeyReaders kr;
      struct{ u32 isKey : 1; u32 isDeleted : 1; i32 readers : 30; };
    };                                                             //  4 bytes  -  kr is key readers  
    u32 idx, version, len, klen, hash;                             // 20 bytes

    BlkLst() : isKey(0), isDeleted(0), readers(0), idx(0), version(0), len(0), klen(0), hash(0) {}
    BlkLst(bool _isKey, i32 _readers, u32 _idx, u32 _version, u32 _len=0, u32 _klen=0, u32 _hash=0) : 
      isKey(_isKey),
      isDeleted(0),
      readers(_readers),
      idx(_idx),
      version(_version),
      hash(_hash)
    {
      len  = _len;
      klen = _klen;
    } 
  };
  struct  BlkCnt { u32 end : 1; u32 cnt : 31; };                                       // this is returned from alloc() and may not be neccesary - it is the number of blocks allocated and if the end was reached

  using ai32        =  std::atomic<i32>;
  using BlockLists  =  lava_vec<BlkLst>;                                               // only the indices returned from the concurrent list are altered, and only one thread will deal with any single index at a time 

  static const u32 LIST_END = CncrLst::LIST_END;

  static VerIdx      List_End()
  { 
    VerIdx vi; 
    vi.idx     = CncrLst::LIST_END; 
    vi.version = 0; 
    return vi; 
  }
  static bool       IsListEnd(VerIdx vi)
  {
    static const VerIdx empty = List_End();
    return empty.asInt == vi.asInt;
  }

  bool           cmpEx(au32* val, u32* expected, u32 desired) const
  {
    using namespace std;
    return atomic_compare_exchange_strong_explicit(
      val, expected, desired,
      memory_order_seq_cst, memory_order_seq_cst
    );
  }
  BlkLst    incReaders(u32 blkIdx) const //u32 version) const                                  // BI is Block Index  increment the readers by one and return the previous kv from the successful swap 
  {
    using namespace std;
    
    KeyReaders cur, nxt;
    BlkLst*     bl  =  &s_bls[blkIdx];
    au32* areaders  =  (au32*)&(bl->kr);
    cur.asInt       =  atomic_load_explicit(areaders, memory_order_seq_cst);
    do{
      if(cur.readers<0 || cur.isDeleted){ return BlkLst(); }
      nxt = cur;
      nxt.readers += 1;
    }while( !cmpEx(areaders, &cur.asInt, nxt.asInt) );

    return *bl;  // after readers has been incremented this block list entry is not going away. The only thing that would change would be the readers and that doesn't matter to the calling function.

    //cur.asInt       =  areaders->load();
    //
    //if(bl->version!=version || cur.readers<0 || cur.isDeleted){ return BlkLst(); }
    //
    //}while( !areaders->compare_exchange_strong(cur.asInt, nxt.asInt) );
  }
  //bool      decReadersOrDel(u32 blkIdx, u32 version, bool del=false) const                   // BI is Block Index  increment the readers by one and return the previous kv from the successful swap 
  bool      decReadersOrDel(u32 blkIdx, bool del=false) const                   // BI is Block Index  increment the readers by one and return the previous kv from the successful swap 
  {
    using namespace std;

    KeyReaders cur, nxt; bool doDelete=false;

    BlkLst*     bl  =  &s_bls[blkIdx];
    au32* areaders  =  (au32*)&(bl->kr);
    cur.asInt       =  atomic_load_explicit(areaders, memory_order_seq_cst);
    do{
      doDelete = false;
      nxt      = cur;
      if(del){
        if(cur.isDeleted){ return true; }
        if(cur.readers==0){
          doDelete      = true; 
        }
        nxt.isDeleted = true;
      }else{
        if(cur.readers==1 &&  cur.isDeleted){ doDelete=true; }
        nxt.readers  -= 1;    
      }
    }while( !cmpEx(areaders, &cur.asInt, nxt.asInt) );
    
    if(doDelete){ doFree(blkIdx); return false; }

    return true;
    
    //cur.asInt       =  areaders->load();
    //if(bl->version!=version){ return false; }
    //
    //if(cur.readers==0 && !cur.isDeleted){ doDelete=true; }
    //
    //}while( !areaders->compare_exchange_strong(cur.asInt, nxt.asInt) );
    //
    //return cur.isDeleted;
  }

//private:
  // s_ variables are used to indicate data structures and memory that is in the shared memory, usually just a pointer on the stack and of course, nothing on the heap
  // The order of the shared memory as it is in the memory mapped file: Version, CncrLst, BlockLists, Blocks
  mutable CncrLst           s_cl;        // flat data structure - pointer to memory 
  mutable BlockLists       s_bls;        // flat data structure - pointer to memory - bl is Block Lists
  void*               s_blksAddr;        // points to the block space in the shared memory
  au64*                s_version;        // pointer to the shared version number

  u32                m_blockSize;
  u64                  m_szBytes;

  VerIdx       nxtBlock(u32  blkIdx)  const
  {
    BlkLst bl  = s_bls[blkIdx];
    prefetch1( (char const* const)blockFreePtr(bl.idx) );
    return VerIdx(bl.idx, bl.version);
  }
  u32     blockFreeSize()             const { return m_blockSize; }
  u8*      blockFreePtr(u32  blkIdx)  const { return ((u8*)s_blksAddr) + blkIdx*m_blockSize; }
  u8*            blkPtr(u32  blkIdx)  const { return ((u8*)s_blksAddr) + blkIdx*m_blockSize; }
  u32      blocksNeeded(u32     len, u32* out_rem=nullptr)
  {
    u32  freeSz   = blockFreeSize();
    u32  byteRem  = len % freeSz;
    u32  blocks   = len / freeSz + (byteRem? 1 : 0);                      // should never be 0 if blocksize is greater than the size of the index type

    if(out_rem) *out_rem = byteRem;

    return blocks;
  }
  u32 findEndSetVersion(u32  blkIdx, u32 version)  const                  // find the last BlkLst slot in the linked list of blocks to free 
  {
    u32 cur=blkIdx, prev=blkIdx;   // the first index will have its version set twice
    while(cur != LIST_END){
      s_bls[cur].version = version;
      prev = cur;
      cur  = s_bls[cur].idx;
    }
    return prev;

    //assert(s_cl.s_lv[cur] == s_bls[cur].idx);
    //
    //sim_assert(s_cl.s_lv[cur]==s_bls[cur].idx, s_cl.s_lv[cur], s_bls[cur].idx );
    //
    //auto lvIdx = s_cl.s_lv[cur];
    //auto blsIdx = s_bls[cur].idx;
    //sim_assert(lvIdx == blsIdx, lvIdx, blsIdx );
    //
    //sim_assert(s_cl.s_lv[prev]==s_bls[prev].idx, s_cl.s_lv[prev], s_bls[prev].idx );
    //
    //return cur;
  }
  void           doFree(u32  blkIdx)  const                                                // frees a list/chain of blocks - don't need to zero out the memory of the blocks or reset any of the BlkLsts' variables since they will be re-initialized anyway
  {
    using namespace std;

    u32 listEnd  =  findEndSetVersion(blkIdx, 0); 


    //sim_assert(s_lv[en], s_lv[en] == LIST_END, en);
    //assert(s_cl.s_lv[listEnd] == LIST_END);

    s_cl.free(blkIdx, listEnd);

    __simdb_deallocs += 1;

    // doesn't work - LIST_END only works for allocation
    //u32 cur = blkIdx;
    //while(cur != LIST_END)
    //  cur = s_cl.free(cur);
  }
  u32        writeBlock(u32  blkIdx, void const* const bytes, u32 len=0, u32 ofst=0)       // don't need to increment readers since write should be done before the block is exposed to any other threads
  {
    u32  blkFree  =  blockFreeSize();
    u8*        p  =  blockFreePtr(blkIdx);
    u32   cpyLen  =  len==0? blkFree : len;                                                // if next is negative, then it will be the length of the bytes in that block
    p      += ofst;
    memcpy(p, bytes, cpyLen);

    return cpyLen;
  }
  u32         readBlock(u32  blkIdx, u32 version, void *const bytes, u32 ofst=0, u32 len=0) const
  {
    (void)version;
    //BlkLst bl = incReaders(blkIdx, version);               
    BlkLst bl = incReaders(blkIdx);
      if(bl.version==0){ return 0; }
      u32   blkFree  =  blockFreeSize();
      u8*         p  =  blockFreePtr(blkIdx);
      u32    cpyLen  =  len==0?  blkFree-ofst  :  len;
      memcpy(bytes, p+ofst, cpyLen);
    decReadersOrDel(blkIdx);
    //decReadersOrDel(blkIdx, version);

    return cpyLen;
  }

public:
  static u64    BlockListsOfst(){ return sizeof(u64); }
  static u64         CListOfst(u32 blockCount){ return BlockListsOfst() + BlockLists::sizeBytes(blockCount); }                 // BlockLists::sizeBytes ends up being sizeof(BlkLst)*blockCount + 2 u64 variables
  static u64          BlksOfst(u32 blockCount){ return CListOfst(blockCount) + CncrLst::sizeBytes(blockCount); }
  static u64         sizeBytes(u32 blockSize, u32 blockCount){ return BlksOfst(blockCount) + blockSize*blockCount; }

  CncrStr(){}
  CncrStr(void* addr, u32 blockSize, u32 blockCount, bool owner=true) :
    s_cl(       (u8*)addr + CListOfst(blockCount), blockCount, owner),
    s_bls(      (u8*)addr + BlockListsOfst(),      blockCount, owner),
    s_blksAddr( (u8*)addr + BlksOfst(blockCount) ),
    s_version(  (au64*)addr ),
    m_blockSize(blockSize),
    m_szBytes( *((u64*)addr) )
  {
    if(owner){
      for(u32 i=0; i<blockCount; ++i){ s_bls[i] = BlkLst(); }
      s_version->store(1);                                                                                   // todo: what is this version for if CncrLst already has a version?
    }
    assert(blockSize > sizeof(i32));
  }

  auto        alloc(u32    size, u32 klen, u32 hash, BlkCnt* out_blocks=nullptr) -> VerIdx    
  {
    u32  byteRem = 0;
    u32   blocks = blocksNeeded(size, &byteRem);
    u32       st = s_cl.alloc(blocks);
    SECTION(handle allocation errors from the concurrent list){
      if(st==LIST_END){
        if(out_blocks){ *out_blocks = {true, 0} ; } 
        return List_End(); 
      }
    }

    u32  ver = (u32)s_version->fetch_add(1);
    u32  cur=st, cnt=0;
    SECTION(loop for the number of blocks needed and get new block and link it to the list)
    {
      for(u32 i=0; i<blocks-1; ++i, ++cnt){
        u32 nxt    = s_cl.s_lv[cur];
        s_bls[cur] = BlkLst(false, 0, nxt, ver, size);
        cur        = nxt;
      }
    }

    SECTION(add the last index into the list, set out_blocks and return the start index with its version)
    {      
      if(out_blocks){
        out_blocks->end = s_cl.s_lv[cur] == LIST_END;
        out_blocks->cnt = cnt;
      }     

      s_bls[cur] = BlkLst(false,0,LIST_END,ver,size,0,0);       // if there is only one block needed, cur and st could be the same

      auto b = s_bls[st]; // debugging

      s_bls[st].isKey = true;
      s_bls[st].hash  = hash;
      s_bls[st].len   = size;
      s_bls[st].klen  = klen;
      s_bls[st].isDeleted = false;

      __simdb_allocs += 1;

      VerIdx vi(st, ver);
      return vi;
    }
  }
  bool         free(u32  blkIdx, u32 version)                                                             // doesn't always free a list/chain of blocks - it decrements the readers and when the readers gets below the value that it started at, only then it is deleted (by the first thread to take it below the starting number)
  {
      (void)version;
    //return decReadersOrDel(blkIdx, version, true);
    return decReadersOrDel(blkIdx, true);
  }
  void          put(u32  blkIdx, void const *const kbytes, u32 klen, void const *const vbytes, u32 vlen)  // don't need version because this will only be used after allocating and therefore will only be seen by one thread until it is inserted into the ConcurrentHash
  {
    using namespace std;
    
    u8*         b  =  (u8*)kbytes;
    bool   kjagged  =  (klen % blockFreeSize()) != 0;
    u32    kblocks  =  kjagged? blocksNeeded(klen)-1 : blocksNeeded(klen);
    u32   remklen   =  klen - (kblocks*blockFreeSize());
    
    u32   fillvlen  =  min(vlen, blockFreeSize()-remklen);
    u32   tailvlen  =  vlen-fillvlen;
    bool   vjagged  =  (tailvlen % blockFreeSize()) != 0;
    u32    vblocks  =  vjagged? blocksNeeded(tailvlen)-1 : blocksNeeded(tailvlen);
    u32    remvlen  =  max<u32>(0, tailvlen - (vblocks*blockFreeSize()) ); 

    u32       cur  =  blkIdx;
    for(u32 i=0; i<kblocks; ++i){
      b   +=  writeBlock(cur, b);
      cur  =  nxtBlock(cur).idx;
    }
    if(kjagged){
      writeBlock(cur, b, remklen);
      b    =  (u8*)vbytes;
      b   +=  writeBlock(cur, b, fillvlen, remklen);
      cur  =  nxtBlock(cur).idx;
    }
    for(u32 i=0; i<vblocks; ++i){
      b   +=  writeBlock(cur, b);
      cur  =  nxtBlock(cur).idx;
    }
    if(vjagged && remvlen>0){
      b   +=  writeBlock(cur, b, remvlen);
    }
  }
  u32           get(u32  blkIdx, u32 version, void *const bytes, u32 maxlen, u32* out_readlen=nullptr) const
  {
    using namespace std;

    if(blkIdx == LIST_END){ return 0; }

    //BlkLst bl = incReaders(blkIdx, version);
    BlkLst bl = incReaders(blkIdx);

    u32 vlen = bl.len-bl.klen;
    if(bl.len==0 || vlen>maxlen ) return 0;

    auto   kdiv = div((i64)bl.klen, (i64)blockFreeSize());
    auto  kblks = kdiv.quot;
    u32    krem = (u32)kdiv.rem;
    auto vrdLen = 0;
    u32     len = 0;
    u32   rdLen = 0;
    u8*       b = (u8*)bytes;
    i32     cur = blkIdx;
    VerIdx  nxt;
    for(int i=0; i<kblks; ++i){ 
      nxt = nxtBlock(cur);           if(nxt.version!=version){ goto read_failure; }
      cur = nxt.idx;
    }

    vrdLen =  min<u32>(blockFreeSize()-krem, vlen);
    rdLen  =  (u32)readBlock(cur, version, b, krem, vrdLen);
    b     +=  rdLen;
    len   +=  rdLen;
    nxt    =  nxtBlock(cur);         if(nxt.version!=version){ goto read_failure; }

    while(len<maxlen && nxt.idx!=LIST_END && nxt.version==version) 
    {
      vrdLen =  min<u32>(blockFreeSize(), maxlen-len);
      cur    =  nxt.idx;
      rdLen  =  readBlock(cur, version, b, 0, vrdLen);  if(rdLen==0) break;        // rdLen is read length
      b     +=  rdLen;
      len   +=  rdLen;
      nxt    =  nxtBlock(cur);
    }

    if(out_readlen){ *out_readlen = len; }

  read_failure:
    decReadersOrDel(blkIdx, false);
    //decReadersOrDel(blkIdx, version);

    return len;                                                                    // only one return after the top to make sure readers can be decremented - maybe it should be wrapped in a struct with a destructor
  }
  u32        getKey(u32  blkIdx, u32 version, void *const bytes, u32 maxlen) const
  {
    if(blkIdx == LIST_END){ return 0; }

    //BlkLst bl = incReaders(blkIdx, version);
    BlkLst bl = incReaders(blkIdx);

    if(bl.len==0 || (bl.klen)>maxlen ) return 0;

    auto   kdiv = div((i64)bl.klen, (i64)blockFreeSize());
    auto  kblks = kdiv.quot;
    u32    krem = (u32)kdiv.rem;
    u32     len = 0;
    u32   rdLen = 0;
    u8*       b = (u8*)bytes;
    VerIdx   vi = { blkIdx, version };

    int i=0;
    while( i<kblks && vi.idx!=LIST_END && vi.version==version) 
    {
      rdLen  =  readBlock(vi.idx, version, b);          if(rdLen==0){ goto read_failure; }     // rdLen is read length
      b     +=  rdLen;
      len   +=  rdLen;
      vi     =  nxtBlock(vi.idx);
      
      ++i;
    }
    rdLen  =  readBlock(vi.idx, version, b, 0, krem);
    b     +=  rdLen;
    len   +=  rdLen;

  read_failure:
    decReadersOrDel(blkIdx);
    //decReadersOrDel(blkIdx, version);

    return len;                                           // only one return after the top to make sure readers can be decremented - maybe it should be wrapped in a struct with a destructor    
  }
  Match   memcmpBlk(u32  blkIdx, u32 version, void const *const buf1, void const *const buf2, u32 len) const    // todo: eventually take out the inc and dec readers and only do them when actually reading and dealing with the whole chain of blocks 
  { 
      (void)blkIdx;
      (void)version;
    // todo: take out inc and dec here, since the whole block list should be read and protected by start of the list
    //if(incReaders(blkIdx, version).len==0){ return MATCH_REMOVED; }
      auto ret = memcmp(buf1, buf2, len);
    //bool freed = !decReadersOrDel(blkIdx, version);

    //if(freed){       return MATCH_REMOVED; }
    //else if(ret==0){ return MATCH_TRUE;    }

    if(ret==0){  return MATCH_TRUE;  }
    else      {  return MATCH_FALSE; }
  }
  Match     compare(u32  blkIdx, u32 version, void const *const buf, u32 len, u32 hash) const
  {
    using namespace std;
    
    BlkLst     bl = s_bls[blkIdx];
    u32 blklstHsh = bl.hash;
    if(blklstHsh!=hash){ return MATCH_FALSE; }                         // vast majority of calls should end here
    bool   verOk  =  bl.version == version;

    u32   curidx  =  blkIdx;
    VerIdx   nxt  =  nxtBlock(curidx);               
    //bool   verOk  =  nxt.version == version;
    //if(nxt.version!=version){ return MATCH_FALSE; }
    
    u32    blksz  =  (u32)blockFreeSize();
    u8*   curbuf  =  (u8*)buf;
    auto    klen  =  s_bls[blkIdx].klen;                            
    if(klen!=len){ return MATCH_FALSE; }
    
    auto  curlen  =  len;
    while(true)
    {
      auto p = blockFreePtr(curidx);
      if(blksz > curlen){
        Match cmpBlk = memcmpBlk(curidx, version, curbuf, p, curlen);                   // the end 
        if(cmpBlk != MATCH_TRUE) return cmpBlk; //MATCH_FALSE;

        return verOk? MATCH_TRUE  :  MATCH_TRUE_WRONG_VERSION;
      }else{
        Match cmp = memcmpBlk(curidx, version, curbuf, p, blksz);   
        if(cmp!=MATCH_TRUE){ return cmp; }
      }

      curbuf  +=  blksz;
      curlen  -=  blksz;
      curidx   =  nxt.idx;
      nxt      =  nxtBlock(curidx);                                 
      
      verOk   &=  nxt.version != version;
      //if(nxt.version!=version){ return MATCH_FALSE; }
    }
  }
  u32           len(u32  blkIdx, u32 version, u32* out_vlen=nullptr) const
  {
    BlkLst bl = s_bls[blkIdx];
    if(version==bl.version && bl.len>0){
      if(out_vlen) *out_vlen = bl.len - bl.klen;
      return bl.len;
    }else 
      return 0;
  }
  auto         list()      const -> CncrLst const& { return s_cl; }
  auto         data()      const -> const void* { return (void*)s_blksAddr; }
  auto       blkLst(u32 i) const -> BlkLst { return s_bls[i]; }

  friend class CncrHsh;
};
class     CncrHsh
{
public:
  using      u8  =   uint8_t;
  using     u32  =  uint32_t;
  using     u64  =  uint64_t;
  using     i64  =   int64_t;
  using    au64  =  std::atomic<u64>;
  using  VerIdx  =  CncrStr::VerIdx;
  using  BlkLst  =  CncrStr::BlkLst;

  struct VerIpd { u32 version, ipd; };                         // ipd is Ideal Position Distance

  static const u32  KEY_MAX          =   0xFFFFFFFF; 
  static const u32  EMPTY            =   KEY_MAX;              // first 21 bits set 
  static const u32  DELETED          =   KEY_MAX - 1;          // 0xFFFFFFFE;       // 1 less than the EMPTY
  static const u32  LIST_END         =   CncrStr::LIST_END;
  static const u32  SLOT_END         =   CncrStr::LIST_END;

  static u64           sizeBytes(u32 size)                   // the size in bytes that this structure will take up in the shared memory
  {
    return lava_vec<VerIdx>::sizeBytes(size) + 16;           // extra 16 bytes for 128 bit alignment padding 
  }
  static u32        nextPowerOf2(u32  v)
  {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
  }
  static u32           HashBytes(const void *const buf, u32 len)
  {
    u64 hsh = fnv_64a_buf(buf, len);
    return (u32)( (hsh>>32) ^ ((u32)hsh));
  }
  static VerIdx         empty_vi(){ return VerIdx(EMPTY,0); }
  static VerIdx       deleted_vi(){ return VerIdx(DELETED,0); }
  static i64              vi_i64(VerIdx vi){ u64 iVi=vi.asInt; return *((i64*)(&iVi)); }                                              // interpret the u64 bits directly as a signed 64 bit integer instead  
  static i64              vi_i64(u64  i){ return *((i64*)&i); }          // interpret the u64 bits directly as a signed 64 bit integer instead    
  static bool            IsEmpty(VerIdx vi)
  {
    static VerIdx emptyvi = empty_vi();
    return emptyvi.asInt == vi.asInt;
  }
  static u32                lo32(u64 n){ return (n>>32); }
  static u32                hi32(u64 n){ return (n<<32)>>32; }
  static u64               swp32(u64 n){ return (((u64)hi32(n))<<32)  |  ((u64)lo32(n)); }
  static u64             inclo32(u64 n, u32 i){ return ((u64)hi32(n)+i)<<32 | lo32(n); }
  static u64             incHi32(u64 n, u32 i){ return ((u64)hi32(n))<<32 | (lo32(n)+i); }
  static u64          shftToHi64(u32 n){ return ((u64)n)<<32; }
  static u64              make64(u32 lo, u32 hi){ return (((u64)lo)<<32) | ((u64)hi); }

private:
  using VerIdxs  =  lava_vec<VerIdx>;

          u32       m_sz;
  mutable VerIdxs   s_vis;                         // s_vis is key value(s) - needs to be changed to versioned indices, m_vis
          CncrStr*  m_csp;                         // csp is concurrent store pointer

  VerIdx       store_vi(u32 i, u64 vi)         const
  {
    using namespace std;
        
    bool odd = i%2 == 1;
    VerIdx strVi;
    if(odd) strVi = VerIdx(lo32(vi), hi32(vi));                                               // the odd numbers need to be swapped so that their indices are on the outer border of 128 bit alignment - the indices need to be on the border of the 128 bit boundary so they can be swapped with an unaligned 64 bit atomic operation
    else    strVi = VerIdx(hi32(vi), lo32(vi));

    u64 prev = atomic_exchange_explicit( (au64*)(s_vis.data()+i), *((u64*)(&strVi)), memory_order_seq_cst);
    //u64 prev = atomic_exchange<u64>( (au64*)(s_vis.data()+i), *((u64*)(&strVi)) );

    if(odd) return VerIdx(lo32(prev), hi32(prev));
    else    return VerIdx(hi32(prev), lo32(prev));
  }
  bool         cmpex_vi(u32 i, VerIdx expected, VerIdx desired) const
  {
    using namespace std;

    u64     exp = i%2? swp32(expected.asInt) : expected.asInt;                                // if the index (i) is odd, swap the upper and lower 32 bits around
    u64    desi = i%2? swp32(desired.asInt) : desired.asInt;                                  // desi is desired int
    au64*  addr = (au64*)(s_vis.data()+i);
    //bool     ok = addr->compare_exchange_strong( exp, desi );
    bool     ok = atomic_compare_exchange_strong_explicit(addr, &exp, desi, memory_order_seq_cst, memory_order_seq_cst);

    return ok;
  }
  //void           doFree(u32 i)                 const
  //{
  //  store_vi(i, empty_vi().asInt);
  //}
  VerIpd            ipd(u32 i, u32 blkIdx)     const                                          // ipd is Ideal Position Distance - it is the distance a CncrHsh index value is from the position that it gets hashed to 
  {
    BlkLst bl = m_csp->blkLst(blkIdx);
    u32    ip = bl.hash % m_sz;                                                               // ip is Ideal Position
    u32   ipd = i>ip?  i-ip  :  m_sz - ip + i;
    return {bl.version, ipd};
  }
  VerIdx           prev(u32 i, u32* out_idx)   const
  {
    *out_idx=prevIdx(i);
    return load(*out_idx);
  }
  VerIdx            nxt(u32 i, u32* out_idx)   const
  {
    *out_idx=nxtIdx(i);
    return load(*out_idx);
  }

  //bool       runIfMatch(VerIdx vi, const void* const buf, u32 len, u32 hash, FUNC f) const 
  //Match       runIfMatch(VerIdx vi, const void* const buf, u32 len, u32 hash, FUNC f) const 
  template<class FUNC, class T>  // Msnh Modified: 2021/05/26
  auto       runIfMatch(VerIdx vi, const void* const buf, u32 len, u32 hash, FUNC f, T defaultRet /*= decltype(f(vi))()*/ ) const -> std::pair<Match, T>   // std::pair<Match, decltype(f(vi))>
  { 
    Match m;
    T funcRet = defaultRet;                                                                   

    //auto b = m_csp->incReaders(vi.idx, vi.version);
    auto b = m_csp->incReaders(vi.idx);
    SECTION(work on the now protected block list without returning until after the readers are decremented)
    {
      if(b.isDeleted){
        m = MATCH_REMOVED;
      }else{
        m = m_csp->compare(vi.idx, vi.version, buf, len, hash);
        if(m==MATCH_TRUE || m==MATCH_TRUE_WRONG_VERSION){
          //funcRet = f(vi); 
          funcRet = f( VerIdx(vi.idx, b.version) );
        }
      }
    }
    //if( !m_csp->decReadersOrDel(vi.idx, vi.version, false) ){ 
   if( !m_csp->decReadersOrDel(vi.idx,false) ){ 
      m = MATCH_REMOVED;
    }

    return {m, funcRet};
    
    // todo: should this increment and decrement the readers, as well as doing something different if it was the thread that freed the blocks
    //
    //if(b.isDeleted){ m = MATCH_REMOVED; } 
    //b.
    //
    //bool matched = false;  
    //decltype(f(vi)) funcRet; // not inside a scope
    //
    //matched=true; 
    //
    //m_csp->decReaders(vi.idx, vi.version);    
    //decReaders(i);
    //
    //return matched;
  }

public:
  CncrHsh(){}
  CncrHsh(void* addr, u32 size, CncrStr* cs, bool owner=true) :
    m_sz(nextPowerOf2(size)),
    m_csp(cs)
  {
    u64     paddr  =  (u64)addr;                // paddr is padded address
    u8        rem  =  16 - paddr%16;
    u8       ofst  =  16 - rem;
    void* algnMem  =  (void*)(paddr+ofst);      assert( ((u64)algnMem) % 16 == 0 ); 

    new (&s_vis) VerIdxs(algnMem, m_sz);        // initialize the lava_vec of VerIdx structs with the 128 bit aligned address
    
    if(owner){
      init(size, cs);
    }
  }
  CncrHsh(CncrHsh const& lval) = delete;
  CncrHsh(CncrHsh&&      rval) = delete;
  CncrHsh& operator=(CncrHsh const& lval) = delete;
  CncrHsh& operator=(CncrHsh&&      rval) = delete;

  VerIdx  operator[](u32 idx) const { return s_vis[idx]; }

  VerIdx   putHashed(u32 hash, VerIdx lstVi, const void *const key, u32 klen) const
  {
    // This function needs to return the VerIdx it was given if there was not a place for the allocation, since it would neighther be stored in the hash map or swapped for another VerIdx that will be freed
    using namespace std;
    static const VerIdx empty   = empty_vi();

    //VerIdx desired = lstVi;
    u32 i=hash%m_sz, en=prevIdx(i);
    for(;; i=nxtIdx(i) )
    {
      VerIdx vi = load(i);
      if(vi.idx>=DELETED){                                                                  // it is either deleted or empty
        bool success = cmpex_vi(i, vi, lstVi);           
        if(success){
          return vi;
        }else{ 
          i=prevIdx(i); 
          continue; 
        }                                                                                   // retry the same loop again if a good slot was found but it was changed by another thread between the load and the compare-exchange
      }                                                                                     // Either we just added the key, or another thread did.

      VerIdx foundVi = empty_vi();
      const auto ths = this;
      auto         f = [ths,i,lstVi,&foundVi](VerIdx vi){
        foundVi      = vi;
        bool success = ths->cmpex_vi(i, vi, lstVi);                                            // this should be hit even when the the versions don't match, since m_csp->compare() will return MATCH_TRUE_WRONG_VERSION
        return success;
      };
      auto cmpAndSuccess = runIfMatch(vi, key, klen, hash, f, false);
      Match          cmp = cmpAndSuccess.first;
      bool       success = cmpAndSuccess.second;

      if(cmp==MATCH_FALSE){
        if(i==en){
          return lstVi;                                                                      // By returning the given VerIdx, we say that there was no place for it found and it needs to be deallocated
        }else{ continue; }
      }else if(cmp==MATCH_REMOVED){                                                          // if the block list is marked as deleted, try this index again, since the index must have changed first
        i=prevIdx(i); 
        continue;
      }

      if(success){ 
        return foundVi;
        //return vi;
      }else{ 
        i=prevIdx(i); 
        continue; 
      }
    }
  }

  template<class FUNC, class T> // Msnh Modified: 2021/05/26
  bool      runMatch(const void *const key, u32 klen, u32 hash, FUNC f, T defaultRet /*= decltype(f(vi))()*/ )       const
  {
    using namespace std;
    
    u32  i = hash % m_sz;
    u32 en = prevIdx(i);
    for(;; i=nxtIdx(i) )
    {
      VerIdx vi = load(i);
      if(vi.idx!=EMPTY && vi.idx!=DELETED){
        Match match = runIfMatch(vi,key,klen,hash,f, defaultRet).first;

        if(match==MATCH_TRUE || match==MATCH_TRUE_WRONG_VERSION){ return true; }
      }

      if(i==en){ return false; }
    }
  }
  
  VerIdx   delHashed(const void *const key, u32 klen, u32 hash)               const
  {  
    using namespace std;
    static const VerIdx   empty = empty_vi();
    static const VerIdx deleted = deleted_vi();

    u32  i = hash % m_sz;
    u32 en = prevIdx(i); 
    for(; i!=en ; i=nxtIdx(i) )
    {
      VerIdx vi = load(i);
      if(vi.idx>=DELETED){continue;}

      Match m = m_csp->compare(vi.idx, vi.version, key, klen, hash);
      if(m==MATCH_TRUE){
        bool success = cmpex_vi(i, vi, deleted);
        if(success){
          //cleanDeletion(i);
          return vi;
        }else{ 
          i=prevIdx(i); continue; 
        }

        //return vi;  // unreachable
      }

      if(m==MATCH_REMOVED || i==en){ return empty; }
    }
    
    return empty;   // not unreachable
  }

  bool          init(u32    sz, CncrStr* cs)
  {
    using namespace std;
        
    m_csp   =  cs;
    m_sz    =  sz;

    for(u32 i=0; i<sz; i+=2) s_vis[i] = VerIdx(EMPTY,0);         // evens 
    for(u32 i=1; i<sz; i+=2) s_vis[i] = VerIdx(0,EMPTY);         // odds
    
    return true;
  }
  VerIdx          at(u32   idx)                const { return load(idx); }
  u32            nxt(u32 stIdx)                const
  {
    auto     idx = stIdx;
    VerIdx empty = empty_vi();
    do{
      VerIdx vi = load(idx);
      if(vi.idx < DELETED){break;}
      idx = (idx+1) % m_sz;                                             // don't increment idx above since break comes before it here

      if(idx==stIdx)
        return SLOT_END;
    }while(true);

    return  idx;
  }
  u32           size()                         const { return m_sz; }
  auto          data()                         const -> void* { return s_vis.data(); }
  u64      sizeBytes()                         const { return s_vis.sizeBytes(); }
  i64            len(const void *const key, u32 klen, u32* out_vlen=nullptr, u32* out_version=nullptr) const
  {
    if(klen<1){return 0;}

    u32 hash=HashBytes(key,klen), i=hash%m_sz, en=prevIdx(i);
    for(;; i=nxtIdx(i) )
    {
      VerIdx vi = load(i);      
      if(vi.idx!=EMPTY && vi.idx!=DELETED){
        if(out_version){ *out_version = vi.version; }
        Match m = m_csp->compare(vi.idx, vi.version, key, klen, hash);
        if(m==MATCH_TRUE){
          return m_csp->len(vi.idx, vi.version, out_vlen);
        }        
      }
      
      if(i==en){ return 0ull; }
    }
  }
  bool           get(const void *const key, u32 klen, void *const out_val, u32 vlen, u32* out_readlen=nullptr) const
  {
    if(klen<1){ return 0; }

    u32 hash=HashBytes(key,klen); 
    CncrStr*  csp = m_csp;
    auto  runFunc = [csp, out_val, vlen, out_readlen](VerIdx vi){
      return csp->get(vi.idx, vi.version, out_val, vlen, out_readlen);
    };

    //Match m = runMatch(key, klen, hash, runFunc, 0);
    return runMatch(key, klen, hash, runFunc, 0);
  }
  bool           put(const void *const key, u32 klen, const void *const val, u32 vlen, u32* out_startBlock=nullptr) 
  {
    assert(klen>0);
    auto dif = __simdb_allocs - __simdb_deallocs;

    u32     hash = CncrHsh::HashBytes(key, klen);
    VerIdx lstVi = m_csp->alloc(klen+vlen, klen, hash);                            // lstVi is block list versioned index
    if(out_startBlock){ *out_startBlock = lstVi.idx; }
    if(lstVi.idx==LIST_END){ 
      return false;
    }

    m_csp->put(lstVi.idx, key, klen, val, vlen);                                  // this writes the data into the blocks before exposing them to other threads through the hash map

    VerIdx vi = putHashed(hash, lstVi, key, klen);                                // put the versioned index in the hash map by swapping it for whatever is there - if there was another index already there, clean it up by freeing it's concurrent list indices and blocks
    if(vi.idx<DELETED){ 
      m_csp->free(vi.idx, vi.version);
    }                                                                             // putHashed returns the entry that was there before, which is the entry that was replaced. If it wasn't empty, we free it here. 
    else{
      auto nxtDif = __simdb_allocs - __simdb_deallocs;
      goto dummy;
      dummy: ;
    }

    //assert(dif == __simdb_allocs - __simdb_deallocs);
    //Println("\nallocs: ", __simdb_allocs, " deallocs: ", __simdb_deallocs);
    //std::cout << std::this_thread::get_id();
    //printf(" allocs: %d  deallocs: %d DIFF: %d\n", __simdb_allocs, __simdb_deallocs, __simdb_allocs - __simdb_deallocs);

    return true;
  }
  bool           del(const void *const key, u32 klen)
  {
    auto     hash = CncrHsh::HashBytes(key, klen);
    VerIdx     vi = delHashed(key, klen, hash);
    bool   doFree = vi.idx<DELETED;
    if(doFree){ m_csp->free(vi.idx, vi.version); }

    return doFree;
  }
  VerIdx        load(u32 i) const
  {    
    assert(i < m_sz);

    au64* avi = (au64*)(s_vis.data()+i);                                           // avi is atomic versioned index
    u64   cur = swp32(avi->load());                                                // need because of endianess? // atomic_load<u64>( (au64*)(m_vis.data()+i) );              // Load the key that was there.

    if(i%2==1) return VerIdx(hi32(cur), lo32(cur));
    else       return VerIdx(lo32(cur), hi32(cur));
  }
  u32         nxtIdx(u32 i) const { return (i+1)%m_sz; }
  u32        prevIdx(u32 i) const { using namespace std; return min(i-1, m_sz-1); }        // clamp to m_sz-1 for the case that hash==0, which will result in an unsigned integer wrap - syntax errors and possible windows min/max macros make this less problematic than std::min() 

};
struct  SharedMem
{
  using    u32  =  uint32_t;
  using    u64  =  uint64_t;
  using   au32  =  std::atomic<u32>;

  static const int alignment = 0;
  
  #ifdef _WIN32
    void*      fileHndl;
  #elif defined(__APPLE__) || defined(__MACH__) || defined(__unix__) || defined(__FreeBSD__) // || defined(__linux__) ?    // osx, linux and freebsd
    int        fileHndl;
  #endif

  void*         hndlPtr;
  void*             ptr;
  u64              size;
  bool            owner;
  char             path[256];

  void mv(SharedMem&& rval)
  {
    fileHndl = rval.fileHndl;
    hndlPtr = rval.hndlPtr;
    ptr = rval.ptr;
    size = rval.size;
    owner = rval.owner;

    strncpy(path, rval.path, sizeof(path));

    rval.clear();
  }

public:
  static void        FreeAnon(SharedMem& sm)
  {
    #ifdef _WIN32
      if(sm.hndlPtr){
        UnmapViewOfFile(sm.hndlPtr);
      }
      if(sm.fileHndl){
        CloseHandle(sm.fileHndl);
      }
    #elif defined(__APPLE__) || defined(__MACH__) || defined(__unix__) || defined(__FreeBSD__) || defined(__linux__)     // osx, linux and freebsd
      if(sm.hndlPtr){
        munmap(sm.hndlPtr, sm.size);  // todo: size here needs to be the total size, and errors need to be checked
      }
      remove(sm.path);
      // todo: deal with errors here as well
    #endif

    sm.clear();
  }
  static SharedMem  AllocAnon(const char* name, u64 sizeBytes, bool raw_path=false, simdb_error* error_code=nullptr)
  {
    using namespace std;

    SharedMem sm;
    sm.hndlPtr  = nullptr;
    sm.owner    = false;
    //sm.size     = alignment==0? sizeBytes  :  alignment-(sizeBytes%alignment);
    sm.size     = sizeBytes;
    if(error_code){ *error_code = simdb_error::NO_ERRORS; }

    #ifdef _WIN32      // windows
      sm.fileHndl = nullptr;
      if(!raw_path){ strcpy(sm.path, "simdb_"); }
    #elif defined(__APPLE__) || defined(__MACH__) || defined(__unix__) || defined(__FreeBSD__) || defined(__linux__)  // osx, linux and freebsd
      sm.fileHndl = 0;
      strcpy(sm.path, P_tmpdir "/simdb_");
    #endif

    u64 len = strlen(sm.path) + strlen(name);
    if(len > sizeof(sm.path)-1){
      *error_code = simdb_error::PATH_TOO_LONG;
      return move(sm);
    }else{ strcat(sm.path, name); }

    #ifdef _WIN32      // windows
      if(raw_path)
      {
        sm.fileHndl = CreateFileA(
          sm.path, 
          GENERIC_READ|GENERIC_WRITE,   //FILE_MAP_READ|FILE_MAP_WRITE,  // apparently FILE_MAP constants have no effects here
          FILE_SHARE_READ|FILE_SHARE_WRITE, 
          NULL,
          CREATE_NEW,
          FILE_ATTRIBUTE_NORMAL,        //_In_ DWORD dwFlagsAndAttributes
          NULL                          //_In_opt_ HANDLE hTemplateFile
        );
      }
      sm.fileHndl = OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, sm.path);

      if(sm.fileHndl==NULL)
      {
        sm.fileHndl = CreateFileMappingA(  // todo: simplify and call this right away, it will open the section if it already exists
          INVALID_HANDLE_VALUE,
          NULL,
          PAGE_READWRITE,
          0,
          (DWORD)sizeBytes,
          sm.path);
        if(sm.fileHndl!=NULL){ sm.owner=true; }
      }
      
      if(sm.fileHndl != nullptr){
        sm.hndlPtr = MapViewOfFile(sm.fileHndl,   // handle to map object
          FILE_MAP_READ | FILE_MAP_WRITE, // FILE_MAP_ALL_ACCESS,   // read/write permission
          0,
          0,
          0);
      }

      if(sm.hndlPtr==nullptr){ 
        int      err = (int)GetLastError();
        LPSTR msgBuf = nullptr;
        /*size_t msgSz =*/ FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                     NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&msgBuf, 0, NULL);
        win_printf("simdb initialization error: %d - %s", err, msgBuf);
        LocalFree(msgBuf);

        CloseHandle(sm.fileHndl); 
        sm.clear(); 
        return move(sm); 
      }
    #elif defined(__APPLE__) || defined(__MACH__) || defined(__unix__) || defined(__FreeBSD__) || defined(__linux__)  // osx, linux and freebsd
      sm.owner  = true; // todo: have to figure out how to detect which process is the owner

      sm.fileHndl = open(sm.path, O_RDWR);
      if(sm.fileHndl == -1)
      {
        sm.fileHndl = open(sm.path, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR |S_IRGRP|S_IWGRP | S_IROTH|S_IWOTH ); // O_CREAT | O_SHLOCK ); // | O_NONBLOCK );
        if(sm.fileHndl == -1){
          if(error_code){ *error_code = simdb_error::COULD_NOT_OPEN_MAP_FILE; }
        }
        else{
          //flock(sm.fileHndl, LOCK_EX);   // exclusive lock  // LOCK_NB
        }
      }else{ sm.owner = false; }

      if(sm.owner){  // todo: still need more concrete race protection?
        fcntl(sm.fileHndl, F_GETLK, &flock);
        flock(sm.fileHndl, LOCK_EX);              // exclusive lock  // LOCK_NB
        //fcntl(sm.fileHndl, F_PREALLOCATE);
        #if defined(__linux__) 
        #else 
          fcntl(sm.fileHndl, F_ALLOCATECONTIG);
        #endif

        if( ftruncate(sm.fileHndl, sizeBytes)!=0 ){
          if(error_code){ *error_code = simdb_error::FTRUNCATE_FAILURE; }
        }
        if( flock(sm.fileHndl, LOCK_UN)!=0 ){
          if(error_code){ *error_code = simdb_error::FLOCK_FAILURE; }
        }
      }

      sm.hndlPtr  = mmap(NULL, sizeBytes, PROT_READ|PROT_WRITE, MAP_SHARED , sm.fileHndl, 0); // MAP_PREFAULT_READ  | MAP_NOSYNC
      close(sm.fileHndl);
      sm.fileHndl = 0;
 
      if(sm.hndlPtr==MAP_FAILED){
        if(error_code){ *error_code = simdb_error::COULD_NOT_MEMORY_MAP_FILE; }
      }
    #endif       
  
    u64      addr = (u64)(sm.hndlPtr);
    u64 alignAddr = addr;
    //if(alignment!=0){ alignAddr = addr + ((alignment-addr%alignment)%alignment); }          // why was the second modulo needed?
    sm.ptr        = (void*)(alignAddr);

    return move(sm);
  }

  SharedMem() :
    hndlPtr(nullptr),
    ptr(nullptr),
    size(0),
    owner(false)
  {}
  SharedMem(SharedMem&)       = delete;
  SharedMem(SharedMem&& rval){ mv(std::move(rval)); }
  SharedMem& operator=(SharedMem&& rval){ mv(std::move(rval)); return *this; }
  ~SharedMem()
  {
    if(ptr){
      au32*   cnt = ((au32*)ptr)+1;
      u64    prev = 0;
      if(cnt->load()>0){ prev = cnt->fetch_sub(1); }
      if(prev==1){ SharedMem::FreeAnon(*this); }
    }
  }
  void clear()
  {
    fileHndl  =  (decltype(fileHndl))0;
    hndlPtr   =  nullptr;
    ptr       =  nullptr;
    size      =  0;
    owner     =  false;
  }
  auto  data() -> void*
  {
    return ptr;
  }
};
class       simdb
{
public:
  using      u8  =  uint8_t;
  using     u32  =  uint32_t;
  using     i32  =   int32_t;
  using     u64  =  uint64_t;
  using     i64  =   int64_t;
  using    au32  =  std::atomic<u32>;
  using    au64  =  std::atomic<u64>;
  using     str  =  std::string;
  using  BlkCnt  =  CncrStr::BlkCnt;
  using  VerIdx  =  CncrHsh::VerIdx;
  using  string  =  std::string;

//private:
  au32*      s_flags;
  au32*      s_cnt;
  au64*      s_blockSize;
  au64*      s_blockCount;
  CncrStr    s_cs;               // store data in blocks and get back indices
  CncrHsh    s_ch;               // store the indices of keys and values - contains a ConcurrentList

  // these variables are local to the stack where simdb lives, unlike the others, they are not simply a pointer into the shared memory
  SharedMem             m_mem;
  mutable simdb_error m_error;
  mutable u32      m_nxtChIdx;
  mutable u32      m_curChIdx;
  u64                m_blkCnt;
  u64                 m_blkSz;
  bool               m_isOpen;

public:
  static const u32        EMPTY = CncrHsh::EMPTY;              // 28 bits set 
  static const u32      DELETED = CncrHsh::DELETED;            // 28 bits set 
  static const u32   FAILED_PUT = CncrHsh::EMPTY;              // 28 bits set 
  static const u32     SLOT_END = CncrHsh::SLOT_END;
  static const u32     LIST_END = CncrStr::LIST_END;

private:
  static u64        OffsetBytes(){ return sizeof(au64)*3; }
  static u64            MemSize(u64 blockSize, u64 blockCount)
  {
    auto  hashbytes = CncrHsh::sizeBytes((u32)blockCount);
    auto storebytes = CncrStr::sizeBytes((u32)blockSize, (u32)blockCount);
    return  hashbytes + storebytes + OffsetBytes();
  }
  static Match     CompareBlock(simdb const *const ths, i32 blkIdx, u32 version, void const *const buf, u32 len, u32 hash)
  { 
    return ths->s_cs.compare(blkIdx, version, buf, len, hash);
  }
  static bool           IsEmpty(VerIdx vi){return CncrHsh::IsEmpty(vi);}         // special value for CncrHsh
  static bool         IsListEnd(VerIdx vi){return CncrStr::IsListEnd(vi);}       // special value for CncrStr

  void mv(simdb&& rval)
  {
    using namespace std;
    
    s_flags      = rval.s_flags;
    s_cnt        = rval.s_cnt;
    s_blockSize  = rval.s_blockSize;
    s_blockCount = rval.s_blockCount;
    memcpy(&s_cs, &rval.s_cs, sizeof(s_cs));
    memcpy(&s_ch, &rval.s_ch, sizeof(s_ch));

    m_mem       =  move(rval.m_mem);
    m_error     =  rval.m_error;
    m_nxtChIdx  =  rval.m_nxtChIdx;
    m_curChIdx  =  rval.m_curChIdx;
    m_blkCnt    =  rval.m_blkCnt;
    m_blkSz     =  rval.m_blkSz;
    m_isOpen    =  rval.m_isOpen;    
  }

public:
  simdb() : 
    m_nxtChIdx(0),
    m_curChIdx(0),
    m_isOpen(false),
    s_flags(nullptr),
    s_cnt(nullptr),
    s_blockSize(nullptr),
    s_blockCount(nullptr)
  {}
  simdb(const char* name, u32 blockSize, u32 blockCount, bool raw_path=false) : 
    m_nxtChIdx(0),
    m_curChIdx(0),
    m_isOpen(false)
  {
    simdb_error error_code = simdb_error::NO_ERRORS;
    new (&m_mem) SharedMem( SharedMem::AllocAnon(name, MemSize(blockSize,blockCount), raw_path, &error_code) );

    if(error_code!=simdb_error::NO_ERRORS){ m_error = error_code; return; }
    if(!m_mem.hndlPtr){ m_error = simdb_error::SHARED_MEMORY_ERROR; return; }

    //  flags     blockSize
    // |----|----|--------|--------|     each dash ('-') represents one byte - flags is the first four, cnt is the next 4, blockSize is the next 8, blockCount is the 8 bytes after that
    //       cnt           blockCount
    s_blockCount  =  ((au64*)m_mem.data())+2;
    s_blockSize   =  ((au64*)m_mem.data())+1;      // 8 byte offset to be after flags and cnt 
    s_flags       =   (au32*)m_mem.data();
    s_cnt         =  ((au32*)m_mem.data())+1;

    if(isOwner()){
      s_blockCount->store(blockCount);
      s_blockSize->store(blockSize);
      s_cnt->store(1);
    }else{
      #if defined(_WIN32)                                          // do we need to spin until ready on windows? unix has file locks built in to the system calls
        //while(s_flags->load()<1){continue;}
      #endif
      s_cnt->fetch_add(1);
      m_mem.size = MemSize(s_blockSize->load(), s_blockCount->load());
    }

    //auto cncrHashSize = CncrHsh::sizeBytes(blockCount);
    uint64_t cncrHashSize = CncrHsh::sizeBytes((u32)s_blockCount->load());
    new (&s_cs) CncrStr( ((u8*)m_mem.data())+cncrHashSize+OffsetBytes(), 
                                 (u32)s_blockSize->load(), 
                                 (u32)s_blockCount->load(), 
                                 m_mem.owner);

    new (&s_ch) CncrHsh( ((u8*)m_mem.data())+OffsetBytes(), 
                                (u32)s_blockCount->load(),
                                &s_cs,                          // the address of the CncrStr
                                m_mem.owner);

    m_blkCnt = s_blockCount->load();
    m_blkSz  = s_blockSize->load();
    m_isOpen = true;

    if(isOwner()){ s_flags->store(1); }
  }
  ~simdb(){ close(); }

  simdb(simdb&& rval){ mv(std::move(rval)); }
  simdb& operator=(simdb&& rval){ mv(std::move(rval)); return *this; }

  i64          len(const void *const key, u32 klen, u32* out_vlen=nullptr, u32* out_version=nullptr) const
  {
    return s_ch.len(key, klen, out_vlen, out_version);
  }
  bool         get(const void *const key, u32 klen, void *const   out_val, u32 vlen, u32* out_readlen=nullptr) const
  {
    return s_ch.get(key, klen, out_val, vlen, out_readlen);
  }
  bool         put(const void *const key, u32 klen, const void *const val, u32 vlen, u32* out_startBlock=nullptr)
  {
    return s_ch.put(key, klen, val, vlen, out_startBlock);
  }
  bool         del(const void *const key, u32 klen){ return s_ch.del(key, klen); }

  i64          len(u32 idx, u32 version, u32* out_klen=nullptr, u32* out_vlen=nullptr) const
  { 
    VerIdx vi = s_ch.load(idx);
    if(vi.idx>=DELETED || vi.version!=version){return 0;}
    u32 total_len = s_cs.len(vi.idx, vi.version, out_vlen); 
    if(total_len>0){
      *out_klen = total_len - *out_vlen;
      return total_len;
    }
    return 0;
  }
  bool         get(char const* const key, void* val, u32 vlen) const
  {
    return get(key, (u32)strlen(key), val, vlen);
  }
  bool         put(char const* const key, const void *const val, u32 vlen, u32* out_startBlock=nullptr)
  {
    assert(m_isOpen); // make sure if the db is being used it has been initialized
    assert(strlen(key)>0);
    return put(key, (u32)strlen(key), val, vlen, out_startBlock);
  }

  void       flush() const
  {
    #ifdef _WIN32
      FlushViewOfFile(m_mem.hndlPtr, m_mem.size);
    #endif
  }
  VerIdx       nxt() const                                                                  // this version index represents a hash index, not an block storage index
  {    
    VerIdx ret = s_ch.empty_vi();
    u32  chNxt = s_ch.nxt(m_nxtChIdx);
    if(chNxt!=SLOT_END){
      m_nxtChIdx = (chNxt + 1) % m_blkCnt;
      ret        = s_ch.at(chNxt);
    }else{
      m_nxtChIdx = (m_nxtChIdx + 1) % m_blkCnt;      
    }
    
    return ret;
  }
  bool      getKey(u32 idx, u32 version, void *const out_buf, u32 klen) const
  {
    if(klen<1) return false;

    VerIdx vi = s_ch.load(idx);  
    if(vi.idx >= CncrHsh::DELETED || vi.version!=version){return false;}
    u32 l = s_cs.getKey(vi.idx, vi.version, out_buf, klen);                               // l is length
    if(l<1){return false;}

    return true;
  }
  u32          cur() const { return m_curChIdx; }
  auto        data() const -> const void* const { return s_cs.data(); }                   // return a pointer to the start of the block data
  u64         size() const { return CncrStr::sizeBytes( (u32)s_blockSize->load(), (u32)s_blockCount->load()); }
  bool     isOwner() const { return m_mem.owner; }
  u64       blocks() const { return s_blockCount->load(); }                               // return the total number of blocks the shared memory
  u64    blockSize() const { return s_blockSize->load();  }
  auto         mem() const -> void* { return m_mem.hndlPtr; }                             // returns a pointer to the start of the shared memory, which will contain the data structures first
  u64      memsize() const { return m_mem.size; }
  auto    hashData() const -> void const* const { return s_ch.data(); }
  bool       close()
  {
    if(m_isOpen){
      m_isOpen = false;
      //u64 prev = s_flags->fetch_sub(1);                                                   // should this be s_cnt? - prev is previous flags value - the number of simdb instances across process that had the shared memory file open
      u64 prev = s_cnt->fetch_sub(1);                                                   // should this be s_cnt? - prev is previous flags value - the number of simdb instances across process that had the shared memory file open
      if(prev==1){                                                                        // if the previous value was 1, that means the value is now 0, and we are the last one to stop using the file, which also means we need to be the one to clean it up
        SharedMem::FreeAnon(m_mem);                                                       // close and delete the shared memory - this is done automatically on windows when all processes are no longer accessing a shared memory file
        return true;
      }
    }
    return false;
  }
  auto       error() const -> simdb_error
  {
    return m_error;
  }

  // separated C++ functions - these won't need to exist if compiled for a C interface
  struct VerStr { 
    u32 ver; string str; 
    bool  operator<(VerStr const& vs) const { return str<vs.str; }
    bool  operator<(string const& rs) const { return str<rs;     }
    bool operator==(VerStr const& vs) const { return str==vs.str && ver==vs.ver; } 
  };   

  i64          len(str    const& key, u32* out_vlen=nullptr, u32* out_version=nullptr) const
  {
    //return len( (void*)key.data(), (u32)key.length(), out_vlen, out_version);
    return len( (void*)key.c_str(), (u32)key.length(), out_vlen, out_version);
  }
  i64          put(str    const& key, str const& value)
  {
    return put(key.data(), (u32)key.length(), value.data(), (u32)value.length());
  }
  bool         get(str    const& key, str*   out_value) const
  {
    u32    vlen = 0;
    len(key.data(), (u32)key.length(), &vlen);
    new (out_value) std::string(vlen,'\0');
    bool     ok = get(key.data(), (u32)key.length(), (void*)out_value->data(), vlen);

    return ok;
  }
  auto         get(str    const& key)                   const -> std::string
  {
    str ret;
    if(this->get(key, &ret)) return ret;
    else return str("");
  }
  VerStr    nxtKey(u64* searched=nullptr)               const
  {
    u32 klen, vlen;
    bool      ok = false;
    i64     prev = (i64)m_nxtChIdx;
    VerIdx viNxt = this->nxt();
    i64     inxt = (i64)m_nxtChIdx;
    u32      cur = s_ch.prevIdx((u32)(inxt));

    if(searched){
      *searched = (inxt-prev-1)>0?  inxt-prev-1  :  (m_blkCnt-prev)+inxt;   //(m_blkCnt-prev-1) + inxt+1;
    }
    if(viNxt.idx>=DELETED){ return {viNxt.version, ""}; }
    
    i64 total_len = this->len(cur, viNxt.version, &klen, &vlen);
    if(total_len==0){ return {viNxt.version, ""}; }
    
    str key(klen,'\0');
    ok         = this->getKey(cur, viNxt.version, 
                              (void*)key.data(), klen); 
                              
    if(!ok || strlen(key.c_str())!=key.length() )
      return {viNxt.version, ""};

    return { viNxt.version, key };                    // copy elision 
  }
  auto  getKeyStrs() const -> std::vector<VerStr>
  {
    using namespace std;
    
    set<VerStr> keys; VerStr nxt; u64 searched=0, srchCnt=0;
    while(srchCnt < m_blkCnt)
    {
      nxt = nxtKey(&searched);
      if(nxt.str.length() > 0){ keys.insert(nxt); }
      
      srchCnt += searched;
    }

    return vector<VerStr>(keys.begin(), keys.end());
  }
  bool         del(str const& key)
  {
    return this->del( (void const* const)key.data(), (u32)key.length() );
  }

  template<class T>
  auto         get(str const& key) -> std::vector<T>
  {
    using namespace std;
    
    u32 vlen = 0;
    //u64  len = len(key.data(), (u32)key.length(), &vlen);
    i64 l = len(key, &vlen); 
    vector<T> ret(vlen);

    u32 readLen = 0;
    bool ok = get(key.data(), (u32)key.length(), (void*)ret.data(), vlen); // &readLen);

    if(ok) return ret;
    else   return vector<T>();
  }
  template<class T>
  i64          put(str    const& key, std::vector<T> const& val)
  {    
    return put(key.data(), (u32)key.length(), val.data(), (u32)(val.size()*sizeof(T)) );
  }
  // end separated C++ functions

};

// simdb_listDBs()
#ifdef _WIN32
  auto simdb_listDBs(simdb_error* error_code=nullptr) -> std::vector<std::string>
  {
    using namespace std;

    static HMODULE                _hModule               = nullptr; 
    static NTOPENDIRECTORYOBJECT  NtOpenDirectoryObject  = nullptr;
    static NTOPENFILE             NtOpenFile             = nullptr;
    static NTQUERYDIRECTORYOBJECT NtQueryDirectoryObject = nullptr;
    static RTLINITUNICODESTRING   RtlInitUnicodeString   = nullptr;
    
    vector<string> ret;

    if(!NtOpenDirectoryObject){  
      //NtOpenDirectoryObject  = (NTOPENDIRECTORYOBJECT)GetLibraryProcAddress( _T("ntdll.dll"), "NtOpenDirectoryObject");
      //NtOpenDirectoryObject  = (NTOPENDIRECTORYOBJECT)GetLibraryProcAddress( (PSTR)_T("ntdll.dll"), (PSTR)_T("NtOpenDirectoryObject") );
      NtOpenDirectoryObject  = (NTOPENDIRECTORYOBJECT)GetLibraryProcAddress( (PSTR)"ntdll.dll", (PSTR)"NtOpenDirectoryObject" );
    }
    if(!NtQueryDirectoryObject){ 
      //NtQueryDirectoryObject = (NTQUERYDIRECTORYOBJECT)GetLibraryProcAddress(_T("ntdll.dll"), "NtQueryDirectoryObject");
      //NtQueryDirectoryObject = (NTQUERYDIRECTORYOBJECT)GetLibraryProcAddress( (PSTR)_T("ntdll.dll"), (PSTR)_T("NtQueryDirectoryObject") );
      NtQueryDirectoryObject = (NTQUERYDIRECTORYOBJECT)GetLibraryProcAddress( (PSTR)"ntdll.dll", (PSTR)"NtQueryDirectoryObject");
    }
    if(!NtOpenFile){ 
      //NtOpenFile = (NTOPENFILE)GetLibraryProcAddress( (PSTR)_T("ntdll.dll"), (PSTR)_T("NtOpenFile") );
      NtOpenFile = (NTOPENFILE)GetLibraryProcAddress( (PSTR)"ntdll.dll", (PSTR)"NtOpenFile" );
    }

    HANDLE     hDir = NULL;
    IO_STATUS_BLOCK  isb = { 0 };
    DWORD sessionId;
    BOOL         ok = ProcessIdToSessionId(GetCurrentProcessId(), &sessionId);
    if(!ok){ return { "Could not get current session" }; }

    wstring     sesspth = L"\\Sessions\\" + to_wstring(sessionId) + L"\\BaseNamedObjects";
    const WCHAR* mempth = sesspth.data();
    
    WCHAR buf[4096];
    UNICODE_STRING pth = { 0 };
    pth.Buffer         = (WCHAR*)mempth;
    pth.Length         = (USHORT)lstrlenW(mempth) * sizeof(WCHAR);
    pth.MaximumLength  = pth.Length;

    OBJECT_ATTRIBUTES oa = { 0 };
    oa.Length             = sizeof( OBJECT_ATTRIBUTES );
    oa.RootDirectory      = NULL;
    oa.Attributes         = OBJ_CASE_INSENSITIVE;                               
    oa.ObjectName         = &pth;
    oa.SecurityDescriptor = NULL;                        
    oa.SecurityQualityOfService = NULL;

    NTSTATUS status;
    status = NtOpenDirectoryObject(
      &hDir, 
      /*STANDARD_RIGHTS_READ |*/ DIRECTORY_QUERY, 
      &oa);

    if(hDir==NULL || status!=STATUS_SUCCESS){ return { "Could not open file" }; }

    BOOLEAN rescan = TRUE;
    ULONG      ctx = 0;
    ULONG   retLen = 0;
    do
    {
      status = NtQueryDirectoryObject(hDir, buf, sizeof(buf), TRUE, rescan, &ctx, &retLen);
      rescan = FALSE;
      auto info = (OBJECT_DIRECTORY_INFORMATION*)buf;

      if( lstrcmpW(info->type.Buffer, L"Section")!=0 ){ continue; }
      WCHAR wPrefix[] = L"simdb_";
      size_t  pfxSz   = sizeof(wPrefix);
      if( strncmp( (char*)info->name.Buffer, (char*)wPrefix, pfxSz)!=0 ){  continue; }

      wstring  wname = wstring( ((WCHAR*)info->name.Buffer)+6 );
      wstring_convert<codecvt_utf8<wchar_t>> cnvrtr;
      string    name = cnvrtr.to_bytes(wname);

      ret.push_back(name);
    }while(status!=STATUS_NO_MORE_ENTRIES);
    
    return ret;
  }
#else
  auto simdb_listDBs(simdb_error* error_code=nullptr) -> std::vector<std::string>
  {
    using namespace std;

    char   prefix[] = "simdb_";
    size_t  pfxSz   = sizeof(prefix)-1;

    vector<string> ret;

    DIR* d;                                          // d is directory handle
    errno = ENOENT;
    if( (d=opendir(P_tmpdir))==NULL || errno!=ENOENT){
      closedir(d);
      if(error_code){ *error_code = simdb_error::DIR_NOT_FOUND; }
      return ret;
    }
 
    struct dirent*     dent;                         // dent is directory entry 
    while( (dent=readdir(d)) != NULL )
    {
      if(errno != ENOENT){
        closedir(d);
        if(error_code){ *error_code = simdb_error::DIR_ENTRY_ERROR; }        
        return ret;
      }

      if(strncmp(dent->d_name, prefix, pfxSz)==0){
        ret.push_back(dent->d_name + 6);
      }
    }

    closedir(d);
    if(error_code){ *error_code = simdb_error::NO_ERRORS; }
    return ret;
  }
#endif


#endif








// return empty;  // should never be reached
//
//Match cmp = runIfMatch(vi, key, klen, hash, f);
//Match cmp = m_csp->compare(vi.idx,vi.version,key,klen,hash);
//bool success = cmpex_vi(i, vi, desired);  // this should be hit even when the the versions don't match, since m_csp->compare() will return MATCH_TRUE_WRONG_VERSION

//u32 cur=blkIdx, prev=blkIdx;   // the first index will have its version set twice
//while(cur != LIST_END){
//  s_bls[prev].version = version;
//  prev = cur;
//  cur  = s_bls[cur].idx;
//}
//return prev;

//auto        alloc(u32    size, u32 klen, u32 hash, BlkCnt* out_blocks=nullptr) -> VerIdx    
//{
//  u32  byteRem = 0;
//  u32   blocks = blocksNeeded(size, &byteRem);
//  u32       st = s_cl.nxt();
//  SECTION(get the starting block index and handle errors)
//  {
//    if(st==LIST_END){
//      if(out_blocks){ *out_blocks = {1, 0} ; } 
//      return List_End(); 
//    }
//  }
//
//  u32  ver = (u32)s_version->fetch_add(1);
//  u32  cur = st;
//  u32  nxt = 0;
//  u32  cnt = 0;
//  SECTION(loop for the number of blocks needed and get new block and link it to the list)
//  {
//    for(u32 i=0; i<blocks-1; ++i)
//    {
//      nxt = s_cl.nxt(cur);
//      if(nxt==LIST_END){ 
//        free(st, ver); 
//        return List_End();
//        //VerIdx empty={LIST_END,0};  // todo: use empty() for this? 
//        //return empty; 
//      } // todo: will this free the start if the start was never set? - will it just reset the blocks but free the index?
//
//      s_bls[cur] = BlkLst(false, 0, nxt, ver, size);
//      //s_cl[cur]  = nxt;
//      cur        = nxt;
//      ++cnt;
//    }
//  }
//
//  SECTION(add the last index into the list, set out_blocks and return the start index with its version)
//  {      
//    s_cl.s_lv[cur] = LIST_END;
//    s_bls[cur] = BlkLst(false,0,LIST_END,ver,size,0,0);       // if there is only one block needed, cur and st could be the same
//
//    auto b = s_bls[st]; // debugging
//
//    s_bls[st].isKey = true;
//    s_bls[st].hash  = hash;
//    s_bls[st].len   = size;
//    s_bls[st].klen  = klen;
//    s_bls[st].isDeleted = false;
//
//    if(out_blocks){
//      out_blocks->end = nxt==LIST_END;
//      out_blocks->cnt = cnt;
//    }     
//    VerIdx vi(st, ver);
//    return vi;
//  }
//}

//s_cl.s_lv[cur] = LIST_END;
//
//u32       st = s_cl.nxt();
//u32  nxt = 0;
//
//nxt = s_cl.nxt(cur);
//if(nxt==LIST_END){ 
//  free(st, ver); 
//  return List_End();
//} // todo: will this free the start if the start was never set? - will it just reset the blocks but free the index?
//
//s_bls[cur] = BlkLst(false, 0, nxt, ver, size);
//cur        = nxt;

//VerIdx empty={LIST_END,0};  // todo: use empty() for this? 
//return empty; 
//
//s_cl[cur]  = nxt;

//u32 findEndSetVersion(u32  blkIdx, u32 version)  const                  // find the last BlkLst slot in the linked list of blocks to free 
//{
//  u32 cur=blkIdx, prev=blkIdx;
//  while(cur != LIST_END){
//    s_bls[prev].version = version;
//    prev = cur;
//    cur  = s_bls[cur].idx;
//  }
//
//  return prev;
//}

//
//u32        prevIdx(u32 i) const { return std::min(i-1, m_sz-1); }        // clamp to m_sz-1 for the case that hash==0, which will result in an unsigned integer wrap
