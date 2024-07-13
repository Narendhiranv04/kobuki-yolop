// Generated by gencpp from file yocs_msgs/LocalizeFeedback.msg
// DO NOT EDIT!


#ifndef YOCS_MSGS_MESSAGE_LOCALIZEFEEDBACK_H
#define YOCS_MSGS_MESSAGE_LOCALIZEFEEDBACK_H


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
struct LocalizeFeedback_
{
  typedef LocalizeFeedback_<ContainerAllocator> Type;

  LocalizeFeedback_()
    : message()  {
    }
  LocalizeFeedback_(const ContainerAllocator& _alloc)
    : message(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _message_type;
  _message_type message;





  typedef boost::shared_ptr< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> const> ConstPtr;

}; // struct LocalizeFeedback_

typedef ::yocs_msgs::LocalizeFeedback_<std::allocator<void> > LocalizeFeedback;

typedef boost::shared_ptr< ::yocs_msgs::LocalizeFeedback > LocalizeFeedbackPtr;
typedef boost::shared_ptr< ::yocs_msgs::LocalizeFeedback const> LocalizeFeedbackConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::yocs_msgs::LocalizeFeedback_<ContainerAllocator1> & lhs, const ::yocs_msgs::LocalizeFeedback_<ContainerAllocator2> & rhs)
{
  return lhs.message == rhs.message;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::yocs_msgs::LocalizeFeedback_<ContainerAllocator1> & lhs, const ::yocs_msgs::LocalizeFeedback_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace yocs_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "5f003d6bcc824cbd51361d66d8e4f76c";
  }

  static const char* value(const ::yocs_msgs::LocalizeFeedback_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x5f003d6bcc824cbdULL;
  static const uint64_t static_value2 = 0x51361d66d8e4f76cULL;
};

template<class ContainerAllocator>
struct DataType< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "yocs_msgs/LocalizeFeedback";
  }

  static const char* value(const ::yocs_msgs::LocalizeFeedback_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n"
"string message\n"
"\n"
;
  }

  static const char* value(const ::yocs_msgs::LocalizeFeedback_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.message);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct LocalizeFeedback_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::yocs_msgs::LocalizeFeedback_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::yocs_msgs::LocalizeFeedback_<ContainerAllocator>& v)
  {
    s << indent << "message: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.message);
  }
};

} // namespace message_operations
} // namespace ros

#endif // YOCS_MSGS_MESSAGE_LOCALIZEFEEDBACK_H
