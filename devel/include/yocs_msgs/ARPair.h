// Generated by gencpp from file yocs_msgs/ARPair.msg
// DO NOT EDIT!


#ifndef YOCS_MSGS_MESSAGE_ARPAIR_H
#define YOCS_MSGS_MESSAGE_ARPAIR_H


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
struct ARPair_
{
  typedef ARPair_<ContainerAllocator> Type;

  ARPair_()
    : left_id(0)
    , right_id(0)
    , baseline(0.0)
    , target_offset(0.0)
    , target_frame()  {
    }
  ARPair_(const ContainerAllocator& _alloc)
    : left_id(0)
    , right_id(0)
    , baseline(0.0)
    , target_offset(0.0)
    , target_frame(_alloc)  {
  (void)_alloc;
    }



   typedef int16_t _left_id_type;
  _left_id_type left_id;

   typedef int16_t _right_id_type;
  _right_id_type right_id;

   typedef float _baseline_type;
  _baseline_type baseline;

   typedef float _target_offset_type;
  _target_offset_type target_offset;

   typedef std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> _target_frame_type;
  _target_frame_type target_frame;





  typedef boost::shared_ptr< ::yocs_msgs::ARPair_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::yocs_msgs::ARPair_<ContainerAllocator> const> ConstPtr;

}; // struct ARPair_

typedef ::yocs_msgs::ARPair_<std::allocator<void> > ARPair;

typedef boost::shared_ptr< ::yocs_msgs::ARPair > ARPairPtr;
typedef boost::shared_ptr< ::yocs_msgs::ARPair const> ARPairConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::yocs_msgs::ARPair_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::yocs_msgs::ARPair_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::yocs_msgs::ARPair_<ContainerAllocator1> & lhs, const ::yocs_msgs::ARPair_<ContainerAllocator2> & rhs)
{
  return lhs.left_id == rhs.left_id &&
    lhs.right_id == rhs.right_id &&
    lhs.baseline == rhs.baseline &&
    lhs.target_offset == rhs.target_offset &&
    lhs.target_frame == rhs.target_frame;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::yocs_msgs::ARPair_<ContainerAllocator1> & lhs, const ::yocs_msgs::ARPair_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace yocs_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::yocs_msgs::ARPair_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::yocs_msgs::ARPair_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::yocs_msgs::ARPair_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::yocs_msgs::ARPair_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::yocs_msgs::ARPair_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::yocs_msgs::ARPair_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::yocs_msgs::ARPair_<ContainerAllocator> >
{
  static const char* value()
  {
    return "9a0e51fbcb2eab37a945707af8ee9a6b";
  }

  static const char* value(const ::yocs_msgs::ARPair_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x9a0e51fbcb2eab37ULL;
  static const uint64_t static_value2 = 0xa945707af8ee9a6bULL;
};

template<class ContainerAllocator>
struct DataType< ::yocs_msgs::ARPair_<ContainerAllocator> >
{
  static const char* value()
  {
    return "yocs_msgs/ARPair";
  }

  static const char* value(const ::yocs_msgs::ARPair_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::yocs_msgs::ARPair_<ContainerAllocator> >
{
  static const char* value()
  {
    return "int16  left_id\n"
"int16  right_id\n"
"float32 baseline\n"
"float32 target_offset\n"
"string  target_frame\n"
;
  }

  static const char* value(const ::yocs_msgs::ARPair_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::yocs_msgs::ARPair_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.left_id);
      stream.next(m.right_id);
      stream.next(m.baseline);
      stream.next(m.target_offset);
      stream.next(m.target_frame);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ARPair_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::yocs_msgs::ARPair_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::yocs_msgs::ARPair_<ContainerAllocator>& v)
  {
    s << indent << "left_id: ";
    Printer<int16_t>::stream(s, indent + "  ", v.left_id);
    s << indent << "right_id: ";
    Printer<int16_t>::stream(s, indent + "  ", v.right_id);
    s << indent << "baseline: ";
    Printer<float>::stream(s, indent + "  ", v.baseline);
    s << indent << "target_offset: ";
    Printer<float>::stream(s, indent + "  ", v.target_offset);
    s << indent << "target_frame: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>::stream(s, indent + "  ", v.target_frame);
  }
};

} // namespace message_operations
} // namespace ros

#endif // YOCS_MSGS_MESSAGE_ARPAIR_H
