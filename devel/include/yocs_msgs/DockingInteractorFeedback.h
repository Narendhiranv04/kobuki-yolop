// Generated by gencpp from file yocs_msgs/DockingInteractorFeedback.msg
// DO NOT EDIT!


#ifndef YOCS_MSGS_MESSAGE_DOCKINGINTERACTORFEEDBACK_H
#define YOCS_MSGS_MESSAGE_DOCKINGINTERACTORFEEDBACK_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace yocs_msgs
{
template <class ContainerAllocator>
struct DockingInteractorFeedback_
{
  typedef DockingInteractorFeedback_<ContainerAllocator> Type;

  DockingInteractorFeedback_()
    : level(0)
    , message()  {
    }
  DockingInteractorFeedback_(const ContainerAllocator& _alloc)
    : level(0)
    , message(_alloc)  {
  (void)_alloc;
    }



   typedef int8_t _level_type;
  _level_type level;

   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _message_type;
  _message_type message;



// reducing the odds to have name collisions with Windows.h 
#if defined(_WIN32) && defined(DEBUG)
  #undef DEBUG
#endif
#if defined(_WIN32) && defined(INFO)
  #undef INFO
#endif
#if defined(_WIN32) && defined(WARN)
  #undef WARN
#endif
#if defined(_WIN32) && defined(ERROR)
  #undef ERROR
#endif

  enum {
    DEBUG = 1,
    INFO = 2,
    WARN = 4,
    ERROR = 8,
  };


  typedef boost::shared_ptr< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> const> ConstPtr;

}; // struct DockingInteractorFeedback_

typedef ::yocs_msgs::DockingInteractorFeedback_<std::allocator<void> > DockingInteractorFeedback;

typedef boost::shared_ptr< ::yocs_msgs::DockingInteractorFeedback > DockingInteractorFeedbackPtr;
typedef boost::shared_ptr< ::yocs_msgs::DockingInteractorFeedback const> DockingInteractorFeedbackConstPtr;

// constants requiring out of line definition

   

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator1> & lhs, const ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator2> & rhs)
{
  return lhs.level == rhs.level &&
    lhs.message == rhs.message;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator1> & lhs, const ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace yocs_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "b042f523c61fe97bdf0d4da2f2289c63";
  }

  static const char* value(const ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xb042f523c61fe97bULL;
  static const uint64_t static_value2 = 0xdf0d4da2f2289c63ULL;
};

template<class ContainerAllocator>
struct DataType< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "yocs_msgs/DockingInteractorFeedback";
  }

  static const char* value(const ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n"
"# Feedback\n"
"int8 level\n"
"\n"
"int8 DEBUG=1\n"
"int8 INFO=2\n"
"int8 WARN=4\n"
"int8 ERROR=8\n"
"\n"
"string message\n"
"\n"
;
  }

  static const char* value(const ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.level);
      stream.next(m.message);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct DockingInteractorFeedback_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::yocs_msgs::DockingInteractorFeedback_<ContainerAllocator>& v)
  {
    s << indent << "level: ";
    Printer<int8_t>::stream(s, indent + "  ", v.level);
    s << indent << "message: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.message);
  }
};

} // namespace message_operations
} // namespace ros

#endif // YOCS_MSGS_MESSAGE_DOCKINGINTERACTORFEEDBACK_H