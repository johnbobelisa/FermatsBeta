import './global.css'
import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Alert,
  Dimensions,
  StatusBar,
  SafeAreaView,
  Platform,
  KeyboardAvoidingView,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import Svg, { Line } from 'react-native-svg';

interface Hold {
  id: string;
  type: 'start' | 'regular' | 'finish';
  position: { x: number; y: number };
}

interface WallScale {
  pixelDistance: string;
  realDistanceCm: string;
}

interface ImageSize {
  width: number;
  height: number;
}

interface Position {
  x: number;
  y: number;
}

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<ImageSize>({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState<ImageSize>({ width: 0, height: 0 });
  const [holds, setHolds] = useState<Hold[]>([]);
  const [selectedMarkerType, setSelectedMarkerType] = useState<'start' | 'regular' | 'finish'>('start');
  const [problemName, setProblemName] = useState<string>('');
  const [wallScale, setWallScale] = useState<WallScale>({ pixelDistance: '', realDistanceCm: '' });
  const [climberHeight, setClimberHeight] = useState<string>('');
  const [climberArmSpan, setClimberArmSpan] = useState<string>('');
  const [isScaleMode, setIsScaleMode] = useState<boolean>(false);
  const [scalePoints, setScalePoints] = useState<Position[]>([]);
  const [scaleFactor, setScaleFactor] = useState<{ x: number; y: number }>({ x: 1, y: 1 });
  
  const holdIdCounter = useRef<number>(0);

  const handleImageUpload = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Sorry, we need camera roll permissions to make this work!');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: 'images',
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      const asset = result.assets[0];
      
      // Calculate display size - use more screen width
      const maxWidth = screenWidth - 32; // 16px padding on each side
      const maxHeight = screenHeight * 0.6; // Use up to 60% of screen height
      const ratio = Math.min(maxWidth / asset.width, maxHeight / asset.height);
      
      const originalWidth = asset.width;
      const originalHeight = asset.height;
      const displayedWidth = originalWidth * ratio;
      const displayedHeight = originalHeight * ratio;

      setImageSize({ width: originalWidth, height: originalHeight });
      setDisplaySize({ width: displayedWidth, height: displayedHeight });
      setScaleFactor({
        x: originalWidth / displayedWidth,
        y: originalHeight / displayedHeight
      });

      setImage(asset.uri);
      setHolds([]);
      setScalePoints([]);
      holdIdCounter.current = 0;
    }
  };

  const getMarkerColor = (type: 'start' | 'regular' | 'finish'): string => {
    switch (type) {
      case 'start': return '#10b981';
      case 'finish': return '#f59e0b';
      case 'regular': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  const validateHolds = (): boolean => {
    const startHolds = holds.filter(h => h.type === 'start');
    const finishHolds = holds.filter(h => h.type === 'finish');
    const regularHolds = holds.filter(h => h.type === 'regular');

    if (startHolds.length < 1 || startHolds.length > 2) {
      Alert.alert('Validation Error', 'You need 1-2 start markers');
      return false;
    }
    if (finishHolds.length !== 1) {
      Alert.alert('Validation Error', 'You need exactly 1 finish marker');      
      return false;
    }
    if (regularHolds.length < 1) {
      Alert.alert('Validation Error', 'You need at least 1 regular marker');
      return false;
    }
    return true;
  };

  const handleImagePress = (event: any) => {
    if (!image) return;

    let x, y;
    
    // Handle web vs mobile differently
    if (Platform.OS === 'web') {
      const rect = event.currentTarget.getBoundingClientRect();
      const clickX = event.clientX - rect.left;
      const clickY = event.clientY - rect.top;
      x = clickX * scaleFactor.x;
      y = clickY * scaleFactor.y;
    } else {
      const { locationX, locationY } = event.nativeEvent;
      x = locationX * scaleFactor.x;
      y = locationY * scaleFactor.y;
    }

    
    if (isScaleMode) {
      const newPoint: Position = { x, y };
      if (scalePoints.length === 0) {
        setScalePoints([newPoint]);
      } else if (scalePoints.length === 1) {
        const point1 = scalePoints[0];
        const distance = Math.sqrt(Math.pow(x - point1.x, 2) + Math.pow(y - point1.y, 2));
        setWallScale({ ...wallScale, pixelDistance: distance.toFixed(0) });
        setScalePoints([point1, newPoint]);
      } else {
        setScalePoints([newPoint]);
        setWallScale({ ...wallScale, pixelDistance: '' });
      }
      return;
    }

    // Check hold limits before adding
    const currentHolds = holds.filter(h => h.type === selectedMarkerType);
    
    if (selectedMarkerType === 'start' && currentHolds.length >= 2) {
      Alert.alert('Hold Limit', 'Maximum 2 start holds allowed');
      return;
    }
    
    if (selectedMarkerType === 'finish' && currentHolds.length >= 1) {
      Alert.alert('Hold Limit', 'Maximum 1 finish hold allowed');
      return;
    }

    const newHold: Hold = {
      id: `h${holdIdCounter.current++}`,
      type: selectedMarkerType,
      position: { x: Math.round(x), y: Math.round(y) }
    };

    setHolds([...holds, newHold]);
  };

  const removeHold = (holdId: string) => {
    setHolds(holds.filter(h => h.id !== holdId));
  };

  const generateBeta = async () => {
    if (!validateHolds()) return;

    if (!problemName || !wallScale.pixelDistance || !wallScale.realDistanceCm || !climberHeight || !climberArmSpan) {
      Alert.alert('Missing Information', 'Please fill in all fields');
      return;
    }

    const payload = {
      problem_name: problemName,
      wall: {
        scale: {
          pixelDistance: parseInt(wallScale.pixelDistance),
          realDistanceCm: parseInt(wallScale.realDistanceCm)
        }
      },
      climber: {
        heightCm: parseInt(climberHeight),
        armSpanCm: parseInt(climberArmSpan)
      },
      holds: holds.map(h => ({
        id: h.id,
        type: h.type,
        position: {
          x: h.position.x,
          y: h.position.y
        }
      }))
    };


    if (!image) {
      Alert.alert('Missing Image', 'Please upload an image.');
      return;
    }

    try {
      const formData = new FormData();

      // 1. Append image as a file
      formData.append('image', {
        uri: image,
        name: 'route.jpg',
        type: 'image/jpeg',
      } as any);

      // 2. Append structured JSON payload as a string
      formData.append('data', JSON.stringify(payload));

      // React Native FormData does not support .entries(), so log manually
      console.log("🧩 Sending Payload:", JSON.stringify(payload, null, 2));
      console.log('📸 Image:', image);

      const response = await fetch('http://192.168.1.111:5000/generate_beta', {
        method: 'POST',
        body: formData,
      });

      const json = await response.json();

      if (response.ok) {
        Alert.alert('Success', 'Beta generated successfully!');
        console.log('Result:', json.result);
      } else {
        Alert.alert('Error', json.error || 'Something went wrong');
      }
    } catch (error) {
      console.error('Error sending data:', error);
      Alert.alert('Network Error', 'Could not connect to server');
    }
  };

  const MarkerTypeButton = ({ type }: { type: 'start' | 'regular' | 'finish' }) => (
    <TouchableOpacity
      onPress={() => setSelectedMarkerType(type)}
      className={`flex-1 py-3 px-4 rounded-xl mx-1 border-2 ${
        selectedMarkerType === type 
          ? 'border-gray-900 shadow-sm' 
          : 'border-gray-200'
      }`}
      style={{ 
        backgroundColor: selectedMarkerType === type ? getMarkerColor(type) : '#ffffff'
      }}
    >
      <Text className={`text-center font-medium capitalize ${
        selectedMarkerType === type ? 'text-white' : 'text-gray-700'
      }`}>
        {type}
      </Text>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView className="flex-1 bg-gray-50">
      <StatusBar barStyle="dark-content" backgroundColor="#f9fafb" />
      
      {/* Clean Header */}
      <View className="bg-white border-b border-gray-100 px-6 py-6">
        <View className="flex-row items-center">
          <View className="w-10 h-10 bg-gray-900 rounded-xl items-center justify-center">
            <Text className="text-white font-bold text-lg">F</Text>
          </View>
          <Text className="ml-3 text-2xl font-bold text-gray-900">
            FermataBeta
          </Text>
        </View>
        <Text className="mt-3 text-gray-600 text-base">
          Generate optimal climbing beta using advanced path optimization
        </Text>
      </View>

      <KeyboardAvoidingView 
        className="flex-1"
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        <ScrollView 
          className="flex-1" 
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
          contentContainerStyle={{ paddingBottom: 40 }}
        >
          {/* Main Content */}
          <View className="p-6">
            {!image ? (
              <View className="mb-8">
                <View className="text-center mb-8">
                  <Text className="text-3xl font-bold text-gray-900 mb-3">Upload Route Image</Text>
                  <Text className="text-gray-600 text-lg leading-relaxed px-4">
                    Start by uploading an image of your climbing route
                  </Text>
                </View>
                
                <TouchableOpacity
                  onPress={handleImageUpload}
                  className="bg-white border-2 border-gray-200 rounded-3xl p-16 items-center mb-8 mx-2 shadow-sm"
                  style={{ borderStyle: 'dashed' }}
                >
                  <View className="w-20 h-20 bg-gray-100 rounded-2xl items-center justify-center mb-6">
                    <Text className="text-3xl">📸</Text>
                  </View>
                  <Text className="text-gray-900 text-xl font-semibold mb-2">
                    Choose Image
                  </Text>
                  <Text className="text-gray-500 text-base">
                    JPG, PNG, or GIF • Max 10MB
                  </Text>
                </TouchableOpacity>
              </View>
            ) : (
              <View className="mb-8">
                <View className="text-center mb-6">
                  <Text className="text-2xl font-bold text-gray-900 mb-2">Mark Your Route</Text>
                  <Text className="text-gray-600 text-base">
                    Tap on holds to mark them, then fill in the details below
                  </Text>
                </View>
                
                <View className="relative bg-white rounded-3xl overflow-hidden shadow-lg border border-gray-100 self-center mb-8">
                  <TouchableOpacity 
                    onPress={handleImagePress} 
                    activeOpacity={1}
                    style={{ position: 'relative' }}
                  >
                    <Image
                      source={{ uri: image }}
                      style={{ width: displaySize.width, height: displaySize.height }}
                      resizeMode="contain"
                    />
                  </TouchableOpacity>
                  
                  {/* Markers */}
                  {holds.map((hold: Hold) => (
                    <TouchableOpacity
                      key={hold.id}
                      onPress={() => removeHold(hold.id)}
                      className="absolute w-5 h-5 rounded-full border-2 border-white shadow-md"
                      style={{
                        backgroundColor: getMarkerColor(hold.type),
                        left: (hold.position.x * displaySize.width) / imageSize.width - 10,
                        top: (hold.position.y * displaySize.height) / imageSize.height - 10,
                      }}
                    />
                  ))}

                  {/* Scale Points */}
                  {scalePoints.map((point, index) => (
                    <View
                      key={`scale-point-${index}`}
                      className="absolute w-6 h-6 bg-amber-500 rounded-full border-2 border-white items-center justify-center shadow-md"
                      style={{
                        left: (point.x * displaySize.width) / imageSize.width - 12,
                        top: (point.y * displaySize.height) / imageSize.height - 12,
                      }}
                    >
                      <Text className="text-white text-xs font-bold">P{index + 1}</Text>
                    </View>
                  ))}
                  
                  {/* Scale line */}
                  {scalePoints.length === 2 && (
                    <Svg 
                      style={{ position: 'absolute', top: 0, left: 0 }}
                      width={displaySize.width} 
                      height={displaySize.height}
                    >
                      <Line
                        x1={(scalePoints[0].x * displaySize.width) / imageSize.width}
                        y1={(scalePoints[0].y * displaySize.height) / imageSize.height}
                        x2={(scalePoints[1].x * displaySize.width) / imageSize.width}
                        y2={(scalePoints[1].y * displaySize.height) / imageSize.height}
                        stroke="#f59e0b"
                        strokeWidth="3"
                        strokeDasharray="5,5"
                      />
                    </Svg>
                  )}
                </View>
                
                <TouchableOpacity
                  onPress={handleImageUpload}
                  className="bg-gray-100 py-4 px-8 rounded-2xl self-center mb-8 border border-gray-200"
                >
                  <Text className="text-gray-700 font-medium">Change Image</Text>
                </TouchableOpacity>
              </View>
            )}

            {image && (
              <View className="space-y-6">
                {/* Scale Setting */}
                <View className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                  <Text className="text-xl font-bold text-gray-900 mb-6">Scale Setting</Text>
                  
                  <TouchableOpacity
                    onPress={() => {
                      setIsScaleMode(!isScaleMode);
                      if (isScaleMode) {
                        setScalePoints([]);
                      }
                    }}
                    className={`py-4 px-6 rounded-xl mb-6 border-2 ${
                      isScaleMode 
                        ? 'bg-amber-500 border-amber-500' 
                        : 'bg-white border-gray-200'
                    }`}
                  >
                    <Text className={`text-center font-semibold ${
                      isScaleMode ? 'text-white' : 'text-gray-900'
                    }`}>
                      {isScaleMode ? '✓ Exit Scale Mode' : '📏 Set Scale (Click 2 points)'}
                    </Text>
                  </TouchableOpacity>
                  
                  <View className="flex-row space-x-4">
                    <View className="flex-1">
                      <Text className="text-gray-700 mb-3 font-semibold">Pixel Distance</Text>
                      <TextInput
                        value={wallScale.pixelDistance}
                        onChangeText={(text) => setWallScale({...wallScale, pixelDistance: text})}
                        placeholder="410"
                        keyboardType="numeric"
                        className="border border-gray-200 rounded-xl px-4 py-4 bg-gray-50 text-base font-medium"
                      />
                    </View>
                    <View className="flex-1">
                      <Text className="text-gray-700 mb-3 font-semibold">Real Distance (cm)</Text>
                      <TextInput
                        value={wallScale.realDistanceCm}
                        onChangeText={(text) => setWallScale({...wallScale, realDistanceCm: text})}
                        placeholder="100"
                        keyboardType="numeric"
                        className="border border-gray-200 rounded-xl px-4 py-4 bg-gray-50 text-base font-medium"
                      />
                    </View>
                  </View>
                </View>

                {/* Marker Controls */}
                {!isScaleMode && (
                  <View className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                    <Text className="text-xl font-bold text-gray-900 mb-6">Hold Type</Text>
                    
                    <View className="flex-row mb-8 mx-1">
                      <MarkerTypeButton type="start" />
                      <MarkerTypeButton type="regular" />
                      <MarkerTypeButton type="finish" />
                    </View>
                    
                    <View className="bg-gray-50 rounded-xl p-6 border border-gray-100">
                      <View className="flex-row justify-around">
                        <View className="items-center">
                          <Text className="font-bold text-emerald-600 text-xl">
                            {holds.filter(h => h.type === 'start').length}/2
                          </Text>
                          <Text className="text-gray-600 text-sm mt-1 font-medium">Start</Text>
                        </View>
                        <View className="items-center">
                          <Text className="font-bold text-blue-600 text-xl">
                            {holds.filter(h => h.type === 'regular').length}
                          </Text>
                          <Text className="text-gray-600 text-sm mt-1 font-medium">Regular</Text>
                        </View>
                        <View className="items-center">
                          <Text className="font-bold text-amber-600 text-xl">
                            {holds.filter(h => h.type === 'finish').length}/1
                          </Text>
                          <Text className="text-gray-600 text-sm mt-1 font-medium">Finish</Text>
                        </View>
                      </View>
                    </View>
                  </View>
                )}

                {/* Problem Details */}
                <View className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                  <Text className="text-xl font-bold text-gray-900 mb-6">Route Details</Text>
                  
                  <View className="mb-6">
                    <Text className="text-gray-700 mb-3 font-semibold">Route Name</Text>
                    <TextInput
                      value={problemName}
                      onChangeText={setProblemName}
                      placeholder="RedV5"
                      className="border border-gray-200 rounded-xl px-4 py-4 bg-gray-50 text-base font-medium"
                    />
                  </View>

                  <View className="flex-row space-x-4">
                    <View className="flex-1">
                      <Text className="text-gray-700 mb-3 font-semibold">Height (cm)</Text>
                      <TextInput
                        value={climberHeight}
                        onChangeText={setClimberHeight}
                        placeholder="170"
                        keyboardType="numeric"
                        className="border border-gray-200 rounded-xl px-4 py-4 bg-gray-50 text-base font-medium"
                      />
                    </View>
                    <View className="flex-1">
                      <Text className="text-gray-700 mb-3 font-semibold">Arm Span (cm)</Text>
                      <TextInput
                        value={climberArmSpan}
                        onChangeText={setClimberArmSpan}
                        placeholder="170"
                        keyboardType="numeric"
                        className="border border-gray-200 rounded-xl px-4 py-4 bg-gray-50 text-base font-medium"
                      />
                    </View>
                  </View>
                </View>

                {/* Generate Beta Button */}
                <View className="mt-8 mb-12">
                  <TouchableOpacity onPress={generateBeta}>
                    <View className="bg-gray-900 py-6 px-8 rounded-2xl shadow-lg">
                      <View className="flex-row items-center justify-center">
                        <Text className="text-2xl mr-3">🚀</Text>
                        <Text className="text-white text-xl font-bold">Generate Beta</Text>
                      </View>
                    </View>
                  </TouchableOpacity>
                </View>
              </View>
            )}
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}